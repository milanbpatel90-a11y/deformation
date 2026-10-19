"use client";
import {Canvas} from "@react-three/fiber";
import {OrbitControls,Environment,useGLTF} from "@react-three/drei";
function Model({url}:{url:string}){const {scene}=useGLTF(url);return <primitive object={scene}/>;}
export default function Viewer({url}:{url:string}){return <div style={{height:560,border:"1px solid #ddd",borderRadius:12,overflow:"hidden"}}><Canvas camera={{position:[0,0,0.35],fov:40}}><ambientLight intensity={0.7}/><directionalLight position={[2,2,3]} intensity={1}/><Environment preset="studio"/><Model url={url}/><OrbitControls/></Canvas></div>}
