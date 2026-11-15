import { AtmosphereSky, atmosphereAerialDepthWgsl } from "@spacexr/atmosphere/webgpu";
import * as cpu from "@spacexr/atmosphere";

/** Independent finite-ray reference: no exposure, palette or image-space correction. */
export async function validateOrbitalAerial(device) {
    const cases=[],failures=[];
    for(const [name,height,angle,sign,multiple] of [
        ["nadir",1e6,0,1,false], ["limb",1e6,59*Math.PI/180,1,false],
        ["night",1e6,0,-1,false], ["far-nadir",1e7,0,1,false],
        ["night-multiple",1e6,0,-1,true],
    ]) {
        const scene=cpu.makePhysicalScene({dustTau:0},[1,0,0]);
        const a=scene.model.ellipsoid.semiMajorAxis,origin=[sign*(a+height),0,0];
        const direction=[-sign*Math.cos(angle),Math.sin(angle),0];
        const camera={positionEcefMeters:origin,forwardEcef:direction,rightEcef:[0,0,-1],
            upEcef:[Math.sin(angle),sign*Math.cos(angle),0],halfWidthTangent:0.02,halfHeightTangent:0.02,viewportWidth:9,viewportHeight:5};
        const sky=await AtmosphereSky.create(device,scene.model,{width:16,hemisphereRows:8,viewSteps:64,solarSteps:192,multipleScattering:multiple});
        const volume=await sky.createAerialVolume({width:9,height:5,depth:17,segmentSteps:2});
        const output=device.createBuffer({size:128,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC});
        const read=device.createBuffer({size:128,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        try {
            const interval=cpu.atmosphericInterval(scene.model,origin,direction);
            const distances=[Math.max(0,interval.start-1000),interval.start,interval.start+0.5*(interval.end-interval.start),interval.end];
            const axes=[a,a,scene.model.ellipsoid.semiMinorAxis];
            const scaled=origin.map((v,c)=>v/axes[c]),ray=direction.map((v,c)=>1000*v/axes[c]);
            const scale=cpu.envelopeScale(scene.model.ellipsoid,scene.model.top);
            const module=device.createShaderModule({code:atmosphereAerialDepthWgsl+`
@group(0)@binding(0)var light:texture_3d<f32>;
@group(0)@binding(1)var trans:texture_3d<f32>;
@group(0)@binding(2)var aerialSampler:sampler;
@group(0)@binding(3)var<storage,read_write>out:array<vec4f>;
@compute @workgroup_size(1)fn main(@builtin(global_invocation_id)id:vec3u){
let distances=array<f32,4>(${distances.map(d=>String(d/1000)).join(',')});
let z=atmosphere_aerialSampleDepth(vec3f(${scaled.join(',')}),vec3f(${ray.join(',')}),${scale},distances[id.x],17.0);
let uv=vec3f(0.5,0.5,z);
out[2u*id.x]=textureSampleLevel(light,aerialSampler,uv,0.0);
out[2u*id.x+1u]=textureSampleLevel(trans,aerialSampler,uv,0.0);}`});
            const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");
            if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
            const pipeline=await device.createComputePipelineAsync({layout:"auto",compute:{module,entryPoint:"main"}});
            sky.update(camera);let encoder;
            for(;;){encoder=device.createCommandEncoder();if(sky.encode(encoder,4096))break;device.queue.submit([encoder.finish()]);await device.queue.onSubmittedWorkDone();}
            volume.encode(encoder,camera);
            const views=volume.bindings();
            const group=device.createBindGroup({layout:pipeline.getBindGroupLayout(0),entries:[
                {binding:0,resource:views.radiance},{binding:1,resource:views.transmission},
                {binding:2,resource:device.createSampler({minFilter:"linear",magFilter:"linear"})},{binding:3,resource:{buffer:output}},
            ]});
            const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,group);pass.dispatchWorkgroups(4);pass.end();
            encoder.copyBufferToBuffer(output,0,read,0,128);device.queue.submit([encoder.finish()]);
            await read.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(read.getMappedRange()));read.unmap();
            const white=cpu.integrateSpectral(scene,origin,direction,{maxDistanceMeters:0}).transmittance;
            const samples=distances.map((distance,i)=>{
                const reference=cpu.integrateSpectral(scene,origin,direction,{maxDistanceMeters:distance,steps:1024,solarSteps:192});
                const light=values.slice(i*8,i*8+3),transmission=values.slice(i*8+4,i*8+7);
                const expectedTransmission=reference.transmittance.map((v,c)=>Math.max(0,Math.min(1,v/white[c])));
                const lightError=Math.max(...light.map((v,c)=>Math.abs(v-Math.max(0,reference.radiance[c]))));
                const transmissionError=Math.max(...transmission.map((v,c)=>Math.abs(v-expectedTransmission[c])));
                if(!Number.isFinite(lightError)||lightError>0.0002)failures.push(`${name} radiance ${i}: ${lightError}`);
                if(!Number.isFinite(transmissionError)||transmissionError>0.002)failures.push(`${name} transmission ${i}: ${transmissionError}`);
                if(i<2&&(light.some(v=>v>1e-6)||transmission.some(v=>Math.abs(v-1)>0.001)))failures.push(`${name}: non-vacuum before air entry`);
                if(sign<0&&light.some(v=>v>1e-6))failures.push(`${name}: sunlight inside the planetary shadow`);
                return {distance,light,transmission,reference:reference.radiance,expectedTransmission,lightError,transmissionError};
            });
            cases.push({name,pathMeters:interval.end-interval.start,samples});
        } finally {await device.queue.onSubmittedWorkDone();volume.dispose();sky.dispose();output.destroy();read.destroy();}
    }
    const near=cases.find(c=>c.name==="nadir").samples[3],far=cases.find(c=>c.name==="far-nadir").samples[3];
    if(Math.max(...near.light.map((v,c)=>Math.abs(v-far.light[c])))>1e-5)failures.push("Vacuum camera distance changes the air column");
    const limb=cases.find(c=>c.name==="limb").samples[3];
    if(limb.light[2]<=near.light[2]||limb.transmission[2]>=near.transmission[2])failures.push("Longer limb path does not increase transport");
    return {cases,failures};
}
