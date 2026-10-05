"use client";

import {useEffect,useRef,useState} from "react";
import {Canvas,useFrame} from "@react-three/fiber";
import {useGLTF} from "@react-three/drei";
import * as THREE from "three";

// MediaPipe's legacy browser package does not ship complete TS declarations.
// The runtime API is stable and loaded only in the browser.
// @ts-ignore
import {FaceMesh} from "@mediapipe/face_mesh";

type Transform={x:number;y:number;scale:number;rotation:number};

function Model({url,transform}:{url:string;transform:Transform}){
  const {scene}=useGLTF(url);
  const group=useRef<THREE.Group>(null);
  const [baseWidth,setBaseWidth]=useState(1);
  useEffect(()=>{
    const box=new THREE.Box3().setFromObject(scene);
    setBaseWidth(Math.max(box.max.x-box.min.x,0.001));
  },[scene]);
  useFrame(()=>{
    if(!group.current)return;
    const targetWidth=Math.max(transform.scale,0.01)*2.3;
    const s=targetWidth/baseWidth;
    group.current.position.set(transform.x,transform.y,0);
    group.current.rotation.z=transform.rotation;
    group.current.scale.setScalar(s);
  });
  return <group ref={group}><primitive object={scene}/></group>;
}

export default function TryOnViewer({url}:{url:string}){
  const videoRef=useRef<HTMLVideoElement>(null);
  const [transform,setTransform]=useState<Transform>({x:0,y:0,scale:0.35,rotation:0});
  const [error,setError]=useState("");
  useEffect(()=>{
    let mesh:any;
    let stream:MediaStream|undefined;
    let cancelled=false;
    async function start(){
      try{
        // @ts-ignore
        mesh=new FaceMesh({locateFile:(file:string)=>`https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`});
        mesh.setOptions({maxNumFaces:1,refineLandmarks:true,minDetectionConfidence:0.6,minTrackingConfidence:0.6});
        mesh.onResults((results:any)=>{
          if(cancelled||!results.multiFaceLandmarks?.length)return;
          const face=results.multiFaceLandmarks[0];
          const left=face[33], right=face[263];
          const cx=(left.x+right.x)/2, cy=(left.y+right.y)/2;
          const dx=right.x-left.x, dy=right.y-left.y;
          const ipd=Math.sqrt(dx*dx+dy*dy);
          setTransform({x:(cx-0.5)*2,y:(0.5-cy)*2,scale:ipd,rotation:-Math.atan2(dy,dx)});
        });
        stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:"user"},audio:false});
        const video=videoRef.current;
        if(!video)return;
        video.srcObject=stream;
        await video.play();
        const loop=async()=>{
          if(cancelled)return;
          if(video.readyState>=2)await mesh.send({image:video});
          requestAnimationFrame(loop);
        };
        loop();
      }catch(e:any){setError(e?.message||"Camera or face tracking could not start.");}
    }
    start();
    return()=>{cancelled=true;stream?.getTracks().forEach(t=>t.stop());mesh?.close?.();};
  },[]);
  return <div style={{position:"relative",height:620,border:"1px solid #ddd",borderRadius:12,overflow:"hidden",background:"#111"}}>
    <video ref={videoRef} playsInline muted style={{position:"absolute",inset:0,width:"100%",height:"100%",objectFit:"cover",transform:"scaleX(-1)"}}/>
    <div style={{position:"absolute",inset:0}}>
      <Canvas orthographic camera={{position:[0,0,10],zoom:1}}>
        <ambientLight intensity={1}/><directionalLight position={[2,2,3]} intensity={1}/>
        <Model url={url} transform={transform}/>
      </Canvas>
    </div>
    {error&&<div style={{position:"absolute",left:16,bottom:16,right:16,padding:12,background:"rgba(0,0,0,.75)",color:"#fff",borderRadius:8}}>{error}</div>}
    <div style={{position:"absolute",top:12,left:12,padding:"6px 10px",background:"rgba(0,0,0,.6)",color:"#fff",borderRadius:6,fontSize:12}}>Face Mesh • IPD aligned</div>
  </div>;
}
