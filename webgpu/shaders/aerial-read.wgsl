// Scaled ECEF origin, ray per kilometre, distance in kilometres.
// Match the visible interval used by the transport integrator, including ground occlusion.
fn atmosphere_aerialInterval(origin:vec3f,ray:vec3f,scale:f32)->vec2f {
    let a=dot(ray,ray);let b=dot(origin,ray);
    let c=dot(origin,origin)-scale*scale;
    let delta=b*b-a*c;if delta<0.0{return vec2f(0.0);}
    let root=sqrt(max(0.0,delta));let start=max(0.0,(-b-root)/a);var end=(-b+root)/a;
    let groundC=dot(origin,origin)-1.0;
    let groundDelta=b*b-a*groundC;
    if groundDelta>=0.0 {
        let ground=(-b-sqrt(groundDelta))/a;
        if ground>=start {end=min(end,ground);}
        else if ground>=-1e-6 && b<0.0 {end=start;}
    }
    return vec2f(start,end);
}
// Continuous depth parameter, with samples concentrated at the dense end of an orbital path.
fn atmosphere_aerialDepth(origin:vec3f,ray:vec3f,scale:f32,distance:f32)->f32 {
    let interval=atmosphere_aerialInterval(origin,ray,scale);
    if interval.y<=interval.x {return 0.0;}
    let f=clamp((distance-interval.x)/(interval.y-interval.x),0.0,1.0);
    return select(sqrt(f),1.0-sqrt(1.0-f),dot(origin,origin)>=scale*scale);
}
// Interpolate cumulative transport in physical distance between the two stored depth slices.
fn atmosphere_aerialSampleDepth(origin:vec3f,ray:vec3f,scale:f32,distance:f32,count:f32)->f32 {
    let z=atmosphere_aerialDepth(origin,ray,scale,distance)*(count-1.0);
    let lo=floor(z);let hi=min(lo+1.0,count-1.0);
    let a=lo/(count-1.0);let b=hi/(count-1.0);let t=z/(count-1.0);
    let outsideAtmosphere=dot(origin,origin)>=scale*scale;
    let fa=select(a*a,a*(2.0-a),outsideAtmosphere);
    let fb=select(b*b,b*(2.0-b),outsideAtmosphere);
    let ft=select(t*t,t*(2.0-t),outsideAtmosphere);
    let weight=select(0.0,(ft-fa)/max(fb-fa,1e-12),hi>lo);
    return (lo+weight+0.5)/count;
}
