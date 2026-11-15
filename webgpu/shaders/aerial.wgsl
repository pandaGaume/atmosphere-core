struct atmosphere_AerialCamera {forward:vec4f,right:vec4f,up:vec4f,size:vec4u}
@group(1) @binding(0) var<uniform> aerialCamera:atmosphere_AerialCamera;
@group(1) @binding(1) var aerialLight:texture_storage_3d<rgba16float,write>;
@group(1) @binding(2) var aerialTransmission:texture_storage_3d<rgba16float,write>;
var<workgroup> aerialXYZ:array<vec3f,128>;
var<workgroup> aerialTransXYZ:array<vec3f,128>;
@compute @workgroup_size(8,1,16)
fn atmosphere_bakeAerial(@builtin(global_invocation_id) id:vec3u,@builtin(local_invocation_id) local:vec3u) {
    let valid=all(id.xy<aerialCamera.size.xy);
    let group=id.z;let groups=arrayLength(&atmosphere_spectrum);
    let spectralRay=valid && group<groups;
    let uv=vec2f(id.xy)/vec2f(aerialCamera.size.xy-vec2u(1));
    let direction=normalize(aerialCamera.forward.xyz+(2.0*uv.x-1.0)*aerialCamera.right.w*aerialCamera.right.xyz+(1.0-2.0*uv.y)*aerialCamera.up.w*aerialCamera.up.xyz);
    let interval=atmosphere_interval(atmosphere_cameraPoint(),direction);
    // Only visible air consumes samples. External observers concentrate them near the surface.
    let outsideAtmosphere=dot(atmosphere_frame.originHi.xyz,atmosphere_frame.originHi.xyz)>=pow(atmosphere_frame.inverseHi.w,2.0);
    var light=vec3f(0.0);var transmission=vec3f(1.0);var whiteXYZ=vec3f(0.0);
    if group==0u {for(var g=0u;g<groups;g++){let band=atmosphere_spectrum[g];whiteXYZ+=atmosphere_toXYZ(band.atmosphere_solar.xyz,band);}}
    let whiteRGB=max(atmosphere_xyzToRgb(whiteXYZ),vec3f(1e-6));
    let slot=local.x*16u+local.z;
    for(var z=0u;z<aerialCamera.size.z;z++) {
        aerialXYZ[slot]=vec3f(0.0);aerialTransXYZ[slot]=vec3f(0.0);
        if spectralRay {
            if z>0u && interval.y>interval.x {
                let za=f32(z-1u)/f32(aerialCamera.size.z-1u);
                let zb=f32(z)/f32(aerialCamera.size.z-1u);
                let fa=select(za*za,za*(2.0-za),outsideAtmosphere);
                let fb=select(zb*zb,zb*(2.0-zb),outsideAtmosphere);
                let start=mix(interval.x,interval.y,fa);
                let end=mix(interval.x,interval.y,fb);
                let result=atmosphere_integrateRange(direction,group,vec2f(start,end),0,i32(aerialCamera.forward.w));
                light+=transmission*result.light;transmission*=result.transmission;
            }
            let band=atmosphere_spectrum[group];
            aerialXYZ[slot]=atmosphere_toXYZ(light*band.atmosphere_solar.xyz,band);
            aerialTransXYZ[slot]=atmosphere_toXYZ(transmission*band.atmosphere_solar.xyz,band);
        }
        workgroupBarrier();
        if valid && group==0u {
            var xyz=vec3f(0.0);var transXYZ=vec3f(0.0);
            for(var g=0u;g<groups;g++){xyz+=aerialXYZ[local.x*16u+g];transXYZ+=aerialTransXYZ[local.x*16u+g];}
            textureStore(aerialLight,vec3i(i32(id.x),i32(id.y),i32(z)),vec4f(max(atmosphere_xyzToRgb(xyz),vec3f(0.0)),1.0));
            textureStore(aerialTransmission,vec3i(i32(id.x),i32(id.y),i32(z)),vec4f(clamp(atmosphere_xyzToRgb(transXYZ)/whiteRGB,vec3f(0.0),vec3f(1.0)),1.0));
        }
        workgroupBarrier();
    }
}

