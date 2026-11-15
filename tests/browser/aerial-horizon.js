import {AtmosphereSky,atmosphereSkyWgsl,atmosphereAerialDepthWgsl} from "@spacexr/atmosphere/webgpu";
import * as cpu from "@spacexr/atmosphere";

/** Nearby opaque surfaces must have continuous transport when their ray crosses the reference ground tangent. */
export async function validateAerialHorizon(device){
    const scene=cpu.makePhysicalScene({dustTau:0},[1,0.5,0]),a=scene.model.ellipsoid.semiMajorAxis,h=2000;
    const origin=[a+h,0,0],angle=-Math.acos(a/(a+h)),forward=[Math.sin(angle),Math.cos(angle),0],up=[Math.cos(angle),-Math.sin(angle),0];
    const camera={positionEcefMeters:origin,forwardEcef:forward,rightEcef:[0,0,-1],upEcef:up,halfWidthTangent:0.02,halfHeightTangent:0.02,viewportWidth:9,viewportHeight:5};
    const sky=await AtmosphereSky.create(device,scene.model,{width:16,hemisphereRows:8,viewSteps:64,solarSteps:192}),volume=await sky.createAerialVolume({width:9,height:5,depth:17,segmentSteps:2});
    const output=device.createBuffer({size:64,usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC}),read=device.createBuffer({size:64,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
    try{
        sky.update(camera);const encoder=device.createCommandEncoder();sky.encode(encoder,4096);volume.encode(encoder,camera);
        const module=device.createShaderModule({code:atmosphereSkyWgsl+atmosphereAerialDepthWgsl+`
@group(1)@binding(0)var light:texture_3d<f32>;@group(1)@binding(1)var transmission:texture_3d<f32>;@group(1)@binding(2)var aerialFilter:sampler;@group(1)@binding(3)var<storage,read_write>out:array<vec4f>;
@compute @workgroup_size(1)fn main(@builtin(global_invocation_id)id:vec3u){let delta=select(-0.00002,0.00002,id.x>0u);let forward=vec3f(${forward.join(',')});let up=vec3f(${up.join(',')});let d=normalize(forward+tan(delta)*up);
let size=vec3f(textureDimensions(light));let uv=vec2f(0.5,0.5-tan(delta)/0.04);
let grid=uv*(size.xy-vec2f(1.0));let lower=floor(grid);let fraction=grid-lower;
var l=vec4f(0.0);var t=vec4f(0.0);
for(var corner=0u;corner<4u;corner++) {
let shift=vec2f(f32(corner&1u),f32((corner>>1u)&1u));let cell=min(lower+shift,size.xy-vec2f(1.0));let sampleUv=cell/(size.xy-vec2f(1.0));
let ray=normalize(forward+(2.0*sampleUv.x-1.0)*0.02*vec3f(0.0,0.0,-1.0)+(1.0-2.0*sampleUv.y)*0.02*up);
let z=atmosphere_aerialSampleDepth(atmosphere_frame.originHi.xyz,ray*atmosphere_frame.inverseHi.xyz,atmosphere_frame.inverseHi.w,35.0,size.z);
let coord=vec3f((cell+vec2f(0.5))/size.xy,z);let weights=select(vec2f(1.0)-fraction,fraction,shift>vec2f(0.0));let weight=weights.x*weights.y;
l+=weight*textureSampleLevel(light,aerialFilter,coord,0.0);t+=weight*textureSampleLevel(transmission,aerialFilter,coord,0.0);
}
out[2u*id.x]=l;out[2u*id.x+1u]=t;}`});
        const errors=(await module.getCompilationInfo()).messages.filter(m=>m.type==="error");if(errors.length)throw new Error(errors.map(m=>m.message).join("\n"));
        const layout=device.createBindGroupLayout({entries:[{binding:0,visibility:GPUShaderStage.COMPUTE,texture:{viewDimension:'3d'}},{binding:1,visibility:GPUShaderStage.COMPUTE,texture:{viewDimension:'3d'}},{binding:2,visibility:GPUShaderStage.COMPUTE,sampler:{type:'filtering'}},{binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}}]});
        const pipeline=await device.createComputePipelineAsync({layout:device.createPipelineLayout({bindGroupLayouts:[sky.bindings().layout,layout]}),compute:{module,entryPoint:'main'}}),views=volume.bindings();
        const group=device.createBindGroup({layout,entries:[{binding:0,resource:views.radiance},{binding:1,resource:views.transmission},{binding:2,resource:device.createSampler({minFilter:'linear',magFilter:'linear'})},{binding:3,resource:{buffer:output}}]});
        const pass=encoder.beginComputePass();pass.setPipeline(pipeline);pass.setBindGroup(0,sky.bindings().group);pass.setBindGroup(1,group);pass.dispatchWorkgroups(2);pass.end();encoder.copyBufferToBuffer(output,0,read,0,64);device.queue.submit([encoder.finish()]);await read.mapAsync(GPUMapMode.READ);const values=Array.from(new Float32Array(read.getMappedRange()));read.unmap();
        const cases=[],failures=[];
        for(let i=0;i<2;i++){const delta=i===0?-0.00002:0.00002,d=cpu.normalize(forward.map((v,c)=>v+Math.tan(delta)*up[c])),reference=cpu.integrateSpectral(scene,origin,d,{maxDistanceMeters:35000,steps:512,solarSteps:192}),light=values.slice(i*8,i*8+3);const error=Math.max(...light.map((v,c)=>Math.abs(v-reference.radiance[c])));if(error>0.004||!Number.isFinite(error))failures.push(`Opaque horizon radiance ${i}: ${error}`);cases.push({delta,light,reference:reference.radiance,error});}
        const jump=Math.max(...cases[0].light.map((v,c)=>Math.abs(v-cases[1].light[c])));if(jump>0.0003)failures.push(`Opaque horizon discontinuity: ${jump}`);
        return{cases,jump,failures};
    }finally{await device.queue.onSubmittedWorkDone();volume.dispose();sky.dispose();output.destroy();read.destroy();}
}
