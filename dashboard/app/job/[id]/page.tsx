"use client";
import {useEffect,useState} from "react";
import {useParams} from "next/navigation";
import dynamic from "next/dynamic";
const Viewer=dynamic(()=>import("../../../components/Viewer"),{ssr:false});
const API=process.env.NEXT_PUBLIC_API||"http://localhost:8000";
export default function Job(){
 const {id}=useParams<{id:string}>(); const [s,setS]=useState<any>({status:"queued",progress:0});
 useEffect(()=>{let timer:any; const poll=async()=>{const r=await fetch(API+"/api/jobs/"+id);const j=await r.json();setS(j);if(j.status!=="done"&&j.status!=="failed")timer=setTimeout(poll,1200)};poll();return()=>clearTimeout(timer)},[id]);
 if(s.status==="failed")return <main style={{padding:32}}><h2>Generation failed</h2><pre>{s.error}</pre></main>;
 if(s.status!=="done")return <main style={{padding:32}}><h2>Generating…</h2><p>{Math.round((s.progress||0)*100)}% — {s.stage||s.status}</p></main>;
 const url=s.glb_url?.startsWith("http")?s.glb_url:API+s.glb_url;
 return <main style={{maxWidth:1100,margin:"30px auto",padding:20}}><h1>Generated glasses</h1><Viewer url={url}/><a href={url} download>Download GLB</a><pre>{JSON.stringify(s.measurements,null,2)}</pre></main>
}
