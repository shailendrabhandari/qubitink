"""Quantum wave-packet interference, version 2. No text appears in frames.

Save quantum_wavepacket.py and run_quantum.py in the same directory.
Install: python -m pip install numpy Pillow moderngl glcontext imageio-ffmpeg
Run: python run_quantum.py --probe
     python run_quantum.py --preview
     python run_quantum.py

Default: 90 seconds, 1080 x 1920, 60 fps, EGL devices 0 and 1.
OpenGL 4.3 compute shaders trace current streamlines directly on each GPU.
GPU rasterisation draws thin antialiased curves and sharp density surfaces.
A small bloom pass leaves the line cores sharp. No titles or watermarks.

Physics: hbar=m=1, anisotropic harmonic trap, omega=(1,sqrt(2),sqrt(3)).
The analytic wavefunction is a normalised superposition of eight coherent
states. This is unitary confined evolution, not a dissipative attractor.
The common zero-point phase is omitted for the display colour convention.
The physical probability current is j=Im(conj(psi)*grad(psi)).

The curves are INSTANTANEOUS streamlines of the current at each frame.
They are integrated with midpoint RK2 in arclength, not physical time.
They are not particle trajectories. Seeds are geometric and do not sample
probability density. Moving highlights indicate curve direction visually,
and do not encode particle speed. Very low-density regions terminate curves.
The faint surfaces are fixed density levels 0.010 and 0.035. Surface lighting,
phase colours, additive line compositing, and camera motion are visual choices.
No Gaussian opacity volume is used. Two GPUs render disjoint chronological
sections of one consistent timeline. The completed MP4 is fully decoded
before success is reported. FFmpeg falls back to imageio-ffmpeg if absent.

--style shells renders the density surfaces without the current curves.
--steps 384 improves streamline integration resolution.
--bitrate 12000 creates a larger local master. Default is 4000 kbps.
--self-test checks analytic norm conservation and density quadrature.
--still 30 saves a PNG. --preview samples film time 24–34, without speeding up.
NVIDIA execution is to be checked locally. Shader execution is tested using
software OpenGL in the build environment.

References:
https://site.physics.georgetown.edu/~jkf/grad_quant2/html/lecture4.html
https://phys.libretexts.org/Bookshelves/Quantum_Mechanics/Essential_Graduate_Physics_-_Quantum_Mechanics_(Likharev)/01%3A_Introduction/1.04%3A_Continuity_Equation
https://moderngl.readthedocs.io/en/latest/reference/compute_shader.html
"""
from __future__ import annotations
import argparse
import concurrent.futures as futures
import json
import multiprocessing as mp
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import numpy as np
from PIL import Image
import moderngl as gl

OMEGA=np.array([1.,np.sqrt(2.),np.sqrt(3.)])
THETA=np.arange(8)*2*np.pi/8
Q0=np.column_stack((1.4*np.cos(THETA),1.4*np.sin(THETA),.8*np.sin(2*THETA)))
P0=np.column_stack((-2.2*np.sin(THETA),2.2*np.cos(THETA),1.3*np.cos(2*THETA)))
ALPHA0=(Q0+1j*P0)/np.sqrt(2.)
RAW=np.exp(1j*(3*THETA+.55*np.sin(3*THETA)))


def gram(alpha):
    norm=np.sum(np.abs(alpha)**2,axis=1)
    return np.exp(-.5*norm[:,None]-.5*norm[None,:]+alpha.conj()@alpha.T)

COEFF=RAW/np.sqrt(np.vdot(RAW,gram(ALPHA0)@RAW).real)
NORMALISATION=float(np.prod(OMEGA/np.pi)**.25)


def state(t):
    alpha=ALPHA0*np.exp(-1j*OMEGA*t)
    return np.sqrt(2)*alpha.real,np.sqrt(2)*alpha.imag


def wavefunction(x,t):
    xi=np.asarray(x)*np.sqrt(OMEGA)
    q,p=state(t)
    result=np.zeros(xi.shape[:-1],dtype=np.complex128)
    for k in range(8):
        delta=xi-q[k]
        phase=xi@p[k]-.5*np.dot(q[k],p[k])
        result+=COEFF[k]*np.exp(-.5*np.sum(delta**2,axis=-1)+1j*phase)
    return NORMALISATION*result*np.exp(-.5j*OMEGA.sum()*t)


def encoder():
    path=shutil.which('ffmpeg')
    if path:return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise RuntimeError('Install FFmpeg with: python -m pip install imageio-ffmpeg') from exc


def self_test():
    for t in [0.,1.,7.,31.]:
        alpha=ALPHA0*np.exp(-1j*OMEGA*t)
        norm=np.vdot(COEFF,gram(alpha)@COEFF)
        if not np.isclose(norm,1,atol=1e-12):raise AssertionError(f'Norm failure: {norm}')
    # Verify the position-space formula against its analytic normalisation.
    x=np.linspace(-7,7,96)
    yy,zz=np.meshgrid(x,x,indexing='ij')
    totals=[]
    for t in [0.,2.7]:
        total=0.
        for a in x:
            coords=np.stack((np.full_like(yy,a),yy,zz),axis=-1)
            total+=np.sum(np.abs(wavefunction(coords,t))**2)
        total*=(x[1]-x[0])**3
        if abs(total-1)>2e-4:raise AssertionError(f'Quadrature norm failure: {total}')
        totals.append(total)
    print('Analytic norm conservation passed. Position-space norms:',totals,flush=True)


FIELD='''
uniform vec3 q[8];uniform vec3 p[8];uniform vec2 coefficient[8];
uniform vec3 frequency_root;uniform float normalisation;
vec3 palette(float phase){
 float a=.5+.5*sin(phase),b=.5+.5*cos(phase);
 vec3 cyan=vec3(.035,.70,1.);
 vec3 violet=vec3(.43,.055,1.);
 vec3 gold=vec3(1.,.37,.025);
 return mix(mix(cyan,violet,b*.65),gold,pow(a,3.)*.8);
}
void field(vec3 r,out float density,out vec3 current,out vec3 gradient,out float phase){
 vec3 xi=r*frequency_root;vec2 value=vec2(0);vec3 gr=vec3(0),gi=vec3(0);
 for(int k=0;k<8;k++){
  vec3 d=xi-q[k];float envelope=exp(-.5*dot(d,d));
  float theta=dot(p[k],xi)-.5*dot(q[k],p[k]);
  vec2 e=vec2(cos(theta),sin(theta));
  vec2 term=envelope*vec2(coefficient[k].x*e.x-coefficient[k].y*e.y,
                          coefficient[k].x*e.y+coefficient[k].y*e.x);
  value+=term;
  gr+=(-d*term.x-p[k]*term.y)*frequency_root;
  gi+=(-d*term.y+p[k]*term.x)*frequency_root;
 }
 float norm2=normalisation*normalisation;
 density=dot(value,value)*norm2;
 current=(value.x*gi-value.y*gr)*norm2;
 gradient=2.*norm2*(value.x*gr+value.y*gi);
 phase=atan(value.y,value.x);
}
'''
COMPUTE='''#version 430
layout(local_size_x=32) in;
struct Vertex { vec4 position; vec4 colour; };
layout(std430,binding=0) buffer Curves { Vertex vertex[]; };
layout(std430,binding=1) readonly buffer Seeds { vec4 seed[]; };
uniform int count;uniform int samples;uniform float arc_step;uniform float film_time;
'''+FIELD+'''
vec3 direction(vec3 r){
 float rho,phase;vec3 current,gradient;field(r,rho,current,gradient,phase);
 return current/max(length(current),1e-12);
}
void main(){
 int line=int(gl_GlobalInvocationID.x);if(line>=count)return;
 vec3 r=seed[line].xyz;bool alive=true;
 for(int i=0;i<512;i++){
  if(i>=samples)break;
  float rho,phase;vec3 current,gradient;field(r,rho,current,gradient,phase);
  if(rho<2e-5 || length(r)>3.45)alive=false;
  float s=float(i)/float(samples-1);
  float fade=smoothstep(0.,.08,s)*(1.-smoothstep(.85,1.,s));
  float depth=clamp(rho/.025,0.,1.);
  float pulse=pow(.5+.5*cos(6.283185*(s-film_time*.18+float(line)*.037)),18.);
  vec3 c=palette(phase)*(.30+.48*depth)*(1.+1.2*pulse)*fade;
  if(!alive)c=vec3(0);
  int at=line*samples+i;
  vertex[at].position=vec4(r,1.);
  vertex[at].colour=vec4(c,1.);
  if(alive){
   // Midpoint RK2 for arclength streamlines of the frozen-time current.
   vec3 v=direction(r);
   vec3 next=direction(r+.5*arc_step*v);
   r+=arc_step*next;
  }
 }
}
'''
LINE_VS='''#version 430
struct Vertex {vec4 position;vec4 colour;};
layout(std430,binding=0) readonly buffer Curves {Vertex vertex[];};
uniform int samples;uniform vec3 camera;uniform mat3 basis;uniform vec2 resolution;
out vec3 vcolour;
void main(){
 int line=gl_VertexID/(2*(samples-1));int segment=(gl_VertexID/2)%(samples-1);
 int endpoint=gl_VertexID%2;int at=line*samples+segment+endpoint;
 vec3 r=transpose(basis)*(vertex[at].position.xyz-camera);
 float z=max(.05,-r.z);vec2 ndc=r.xy/(z*.43);
 ndc.x/=resolution.x/resolution.y;
 gl_Position=vec4(ndc,0.,1.);
 float depth=clamp((length(camera)-z+3.)/6.,0.,1.);
 vcolour=vertex[at].colour.rgb*(.4+.6*depth);
}
'''
LINE_GS='''#version 430
layout(lines) in;layout(triangle_strip,max_vertices=4) out;
in vec3 vcolour[];out vec3 colour;out float lateral;
uniform vec2 resolution;uniform float width;
void main(){
 vec2 a=gl_in[0].gl_Position.xy,b=gl_in[1].gl_Position.xy;
 vec2 delta=(b-a)*resolution;
 vec2 offset=normalize(vec2(-delta.y,delta.x)+vec2(1e-9))*width/resolution;
 colour=vcolour[0];lateral=-1.;gl_Position=vec4(a-offset,0,1);EmitVertex();
 colour=vcolour[0];lateral=1.;gl_Position=vec4(a+offset,0,1);EmitVertex();
 colour=vcolour[1];lateral=-1.;gl_Position=vec4(b-offset,0,1);EmitVertex();
 colour=vcolour[1];lateral=1.;gl_Position=vec4(b+offset,0,1);EmitVertex();EndPrimitive();
}
'''
LINE_FS='''#version 430
in vec3 colour;in float lateral;out vec4 frag;
void main(){float alpha=1.-smoothstep(.45,1.,abs(lateral));frag=vec4(colour*alpha,1);}
'''
VS='''#version 430
in vec2 position;out vec2 uv;
void main(){uv=position*.5+.5;gl_Position=vec4(position,0,1);}
'''
SHELL='''#version 430
in vec2 uv;out vec4 frag;
uniform vec2 resolution;uniform vec3 camera;uniform mat3 basis;uniform float strength;
'''+FIELD+'''
void main(){
 vec2 xy=(uv-.5)*2.;xy.x*=resolution.x/resolution.y;
 vec3 ray=normalize(basis*vec3(xy*.43,-1.));
 float b=dot(camera,ray),c=dot(camera,camera)-27.04;
 float disc=b*b-c;if(disc<0.){frag=vec4(0,0,0,1);return;}
 float lo=max(0.,-b-sqrt(disc)),hi=-b+sqrt(disc),ds=(hi-lo)/128.;
 vec3 colour=vec3(0);float previous,phase;vec3 current,gradient;
 field(camera+ray*lo,previous,current,gradient,phase);
 int hits=0;
 for(int i=0;i<128;i++){
  float t1=lo+(float(i)+1.)*ds;
  float rho;field(camera+ray*t1,rho,current,gradient,phase);
  for(int shell=0;shell<2;shell++){
   float level=shell==0?.010:.035;
   if((previous-level)*(rho-level)<0.){
    float a=t1-ds,z=t1,fa=previous-level;
    for(int refine=0;refine<7;refine++){
     float mid=(a+z)*.5,r;field(camera+ray*mid,r,current,gradient,phase);
     if(fa*(r-level)<=0.)z=mid;else{a=mid;fa=r-level;}
    }
    float r;field(camera+ray*((a+z)*.5),r,current,gradient,phase);
    vec3 normal=normalize(-gradient+vec3(1e-10));
    float rim=pow(1.-abs(dot(normal,ray)),3.);
    float light=max(0.,dot(normal,normalize(vec3(-.7,.8,1.))));
    colour+=palette(phase)*strength*(.06+.30*light+1.4*rim);
    hits++;
   }
  }
  previous=rho;if(hits>10)break;
 }
 frag=vec4(colour,1);
}
'''
BLUR='''#version 430
in vec2 uv;out vec4 frag;uniform sampler2D source;uniform vec2 delta;
void main(){
 vec3 c=texture(source,uv).rgb*.227027;
 c+=(texture(source,uv+delta*1.384615).rgb+texture(source,uv-delta*1.384615).rgb)*.316216;
 c+=(texture(source,uv+delta*3.230769).rgb+texture(source,uv-delta*3.230769).rgb)*.070270;
 frag=vec4(c,1);
}
'''
COMPOSE='''#version 430
in vec2 uv;out vec4 frag;uniform sampler2D core;uniform sampler2D bloom;uniform float fade;
void main(){
 vec3 c=texture(core,uv).rgb+texture(bloom,uv).rgb*.16;
 c=pow(1.-exp(-c*1.7),vec3(1./1.5));
 frag=vec4((vec3(.004,.007,.014)+c)*fade,1);
}
'''


def smooth(a,b,t):
    u=np.clip((t-a)/(b-a),0.,1.)
    return float(u*u*(3-2*u))


class Renderer:
    def __init__(self,device,w,h,steps=256,allow_software=False,style='filaments'):
        self.w,self.h,self.steps,self.style=w,h,steps,style
        self.count=128
        self.ctx=gl.create_standalone_context(require=430,backend='egl',device_index=device)
        name=self.ctx.info['GL_RENDERER']
        print(f'EGL device {device}: {name}',flush=True)
        if not allow_software and any(s in name.lower() for s in ['llvmpipe','softpipe','swrast']):
            self.ctx.release();raise RuntimeError('Select an NVIDIA EGL device using --probe.')
        ctx=self.ctx
        self.compute=ctx.compute_shader(COMPUTE)
        self.lines=ctx.program(vertex_shader=LINE_VS,geometry_shader=LINE_GS,fragment_shader=LINE_FS)
        self.shell=ctx.program(vertex_shader=VS,fragment_shader=SHELL)
        self.blur=ctx.program(vertex_shader=VS,fragment_shader=BLUR)
        self.compose=ctx.program(vertex_shader=VS,fragment_shader=COMPOSE)
        self.curves=ctx.buffer(reserve=self.count*steps*8*4)
        self.curves.bind_to_storage_buffer(0)
        # A deterministic set of seeds. They do not claim to sample |psi|^2.
        seed=[]
        for z in np.linspace(-1.3,1.3,8):
            for j in range(16):
                angle=2*np.pi*j/16+.25*z
                radius=1.1+.55*np.cos(z*1.1)+.12*np.sin(3*angle)
                seed.append([radius*np.cos(angle),radius*np.sin(angle),z,1.])
        self.seeds=ctx.buffer(np.array(seed,dtype='f4').tobytes())
        self.seeds.bind_to_storage_buffer(1)
        self.compute['count'].value=self.count;self.compute['samples'].value=steps
        self.compute['arc_step'].value=6.0/(steps-1)
        self.lines['samples'].value=steps;self.lines['resolution'].value=(w,h)
        self.lines['width'].value=max(1.4,w/1080*2.0)
        self.line_vao=ctx.vertex_array(self.lines,[])
        quad=ctx.buffer(np.array([-1,-1,1,-1,-1,1,1,1],dtype='f4').tobytes())
        self.vaos=[ctx.vertex_array(shader,[(quad,'2f','position')])
                   for shader in [self.shell,self.blur,self.compose]]
        self.tex=[];self.fbos=[]
        for size,dtype in [((w,h),'f2'),((w//2,h//2),'f2'),((w//2,h//2),'f2'),((w,h),'f1')]:
            tex=ctx.texture(size,3,dtype=dtype);tex.filter=(gl.LINEAR,gl.LINEAR)
            tex.repeat_x=False;tex.repeat_y=False
            self.tex.append(tex);self.fbos.append(ctx.framebuffer([tex]))
        for shader in [self.compute,self.shell]:
            shader['frequency_root'].value=tuple(np.sqrt(OMEGA))
            shader['normalisation'].value=NORMALISATION
            shader['coefficient'].write(np.column_stack((COEFF.real,COEFF.imag)).astype('f4').tobytes())
        self.shell['resolution'].value=(w,h)
        self.shell['strength'].value=.045 if style=='filaments' else .5
        self.blur['source'].value=0
        self.compose['core'].value=0;self.compose['bloom'].value=1

    def frame(self,t):
        q,p=state(.28*t)
        for shader in [self.compute,self.shell]:
            shader['q'].write(q.astype('f4').tobytes());shader['p'].write(p.astype('f4').tobytes())
        self.compute['film_time'].value=t
        yaw=.026*t+.35;pitch=.18+.10*np.sin(t*.045)
        distance=14.5-.4*smooth(20,40,t)+.4*smooth(68,82,t)
        camera=distance*np.array([np.cos(pitch)*np.sin(yaw),np.sin(pitch),np.cos(pitch)*np.cos(yaw)])
        forward=-camera/np.linalg.norm(camera)
        right=np.cross(forward,[0.,1.,0.]);right/=np.linalg.norm(right)
        up=np.cross(right,forward);basis=np.column_stack((right,up,-forward))
        for shader in [self.shell,self.lines]:
            shader['camera'].value=tuple(camera);shader['basis'].write(basis.astype('f4').T.tobytes())
        ctx=self.ctx
        self.fbos[0].use();ctx.viewport=(0,0,self.w,self.h);ctx.clear()
        self.vaos[0].render(gl.TRIANGLE_STRIP)
        if self.style=='filaments':
            self.compute.run(group_x=(self.count+31)//32)
            ctx.memory_barrier()
            ctx.enable(gl.BLEND);ctx.blend_func=(gl.ONE,gl.ONE)
            self.line_vao.render(gl.LINES,vertices=self.count*(self.steps-1)*2)
            ctx.disable(gl.BLEND)
        source=self.tex[0]
        for i in range(2):
            target=1+i%2
            self.fbos[target].use();ctx.viewport=(0,0,self.w//2,self.h//2)
            source.use(0)
            self.blur['delta'].value=(2/self.w,0) if i%2==0 else (0,2/self.h)
            self.vaos[1].render(gl.TRIANGLE_STRIP);source=self.tex[target]
        self.fbos[3].use();ctx.viewport=(0,0,self.w,self.h)
        self.tex[0].use(0);source.use(1)
        self.compose['fade'].value=min(smooth(0,1.5,t),1-smooth(88,90,t))
        self.vaos[2].render(gl.TRIANGLE_STRIP)
        raw=self.fbos[3].read(components=3,alignment=1)
        return Image.frombytes('RGB',(self.w,self.h),raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)


def validate(path):
    subprocess.run([encoder(),'-v','error','-xerror','-i',str(path),'-f','null','-'],check=True)


def render_part(job):
    device,w,h,fps,steps,start,nframes,path,bitrate,style=job
    movie=Renderer(device,w,h,steps,style=style)
    cmd=[encoder(),'-y','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
         '-s',f'{w}x{h}','-r',str(fps),'-i','-','-an','-c:v','libx264',
         '-preset','fast','-b:v',f'{bitrate}k','-maxrate',f'{int(bitrate*1.15)}k',
         '-bufsize',f'{bitrate*2}k','-pix_fmt','yuv420p',str(path)]
    proc=subprocess.Popen(cmd,stdin=subprocess.PIPE)
    try:
        for i in range(nframes):
            proc.stdin.write(movie.frame(start+i/fps).tobytes())
            if i%(fps*5)==0:print(f'Device {device}: {i/fps:.0f}/{nframes/fps:.0f}s in section {start:g}s',flush=True)
    finally:
        proc.stdin.close()
    if proc.wait():raise RuntimeError('Encoder failed.')
    validate(path)
    return str(path)


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gpus',default='0,1')
    ap.add_argument('--probe',action='store_true')
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--preview',action='store_true',help='10 seconds from film time 24–34 seconds')
    ap.add_argument('--still',type=float,help='One PNG at a specified film time')
    ap.add_argument('--width',type=int,default=1080)
    ap.add_argument('--height',type=int,default=1920)
    ap.add_argument('--fps',type=int,default=60)
    ap.add_argument('--steps',type=int,default=256,help='Samples per current streamline, 32–512')
    ap.add_argument('--style',choices=['filaments','shells'],default='filaments')
    ap.add_argument('--bitrate',type=int,default=4000,help='H.264 kbps; use 12000 or higher for a local master')
    ap.add_argument('--output',type=Path)
    args=ap.parse_args(argv)
    if args.self_test:self_test();return
    try:devices=[int(x) for x in args.gpus.split(',')]
    except ValueError:ap.error('GPU indices must be integers separated by commas.')
    if not devices or min(devices)<0 or len(devices)!=len(set(devices)):ap.error('Use unique nonnegative EGL indices.')
    if args.probe:
        for device in devices:
            try:
                ctx=gl.create_standalone_context(require=430,backend='egl',device_index=device)
                print(device,ctx.info['GL_VENDOR'],ctx.info['GL_RENDERER']);ctx.release()
            except Exception as exc:print(device,str(exc))
        return
    if min(args.width,args.height,args.fps,args.bitrate)<1 or args.width%2 or args.height%2:
        ap.error('Dimensions must be positive and even. FPS and bitrate must be positive.')
    if not 32<=args.steps<=512:ap.error('Steps must lie between 32 and 512.')
    if args.still is not None:
        if not 0<=args.still<=90:ap.error('Still time must lie between 0 and 90 seconds.')
        path=(args.output or Path('quantum_wavepacket.png')).resolve().with_suffix('.png')
        path.parent.mkdir(parents=True,exist_ok=True)
        Renderer(devices[0],args.width,args.height,args.steps,style=args.style).frame(args.still).save(path)
        print(path);return
    encoder()
    output=(args.output or Path('quantum_preview.mp4' if args.preview else 'quantum_wavepacket_90s.mp4')).resolve()
    if output.suffix.lower()!='.mp4':ap.error('Video output must have an .mp4 extension.')
    output.parent.mkdir(parents=True,exist_ok=True)
    start,duration=(24,10) if args.preview else (0,90)
    # Probe every requested device before starting an expensive render.
    for device in devices:
        ctx=gl.create_standalone_context(require=430,backend='egl',device_index=device)
        name=ctx.info['GL_RENDERER'];ctx.release()
        if any(s in name.lower() for s in ['llvmpipe','softpipe','swrast']):
            raise RuntimeError(f'Device {device} uses {name}. Choose an NVIDIA EGL device.')
    began=time.monotonic()
    with tempfile.TemporaryDirectory(prefix='quantum_sections_',dir=output.parent) as tmp:
        folder=Path(tmp);jobs=[];cursor=0;total=duration*args.fps
        for i,device in enumerate(devices):
            frames=total//len(devices)+(i<total%len(devices))
            if not frames:continue
            jobs.append((device,args.width,args.height,args.fps,args.steps,start+cursor/args.fps,
                         frames,folder/f'section_{i:02d}.mp4',args.bitrate,args.style))
            cursor+=frames
        with futures.ProcessPoolExecutor(max_workers=len(jobs),mp_context=mp.get_context('spawn')) as pool:
            paths=list(pool.map(render_part,jobs))
        listing=folder/'sections.txt'
        listing.write_text(''.join(f"file '{Path(p).name}'\n" for p in paths))
        subprocess.run([encoder(),'-y','-loglevel','error','-f','concat','-safe','0','-i',str(listing),
                        '-c','copy','-movflags','+faststart',str(output)],check=True)
    validate(output)
    output.with_suffix('.json').write_text(json.dumps(dict(duration=duration,width=args.width,
        height=args.height,fps=args.fps,steps=args.steps,style=args.style,omega=OMEGA.tolist(),
        model='normalised coherent-state superposition in a 3D anisotropic harmonic trap',
        model_time_per_video_second=.28,text_in_frames=False,phase_gauge='common zero-point phase removed'),indent=2))
    print(f'Finished and fully decoded: {output}\nElapsed: {time.monotonic()-began:.0f}s',flush=True)


if __name__=='__main__':main()
