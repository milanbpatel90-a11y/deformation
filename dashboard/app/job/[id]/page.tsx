"use client";

import {useEffect,useState} from "react";
import {useParams} from "next/navigation";
import dynamic from "next/dynamic";
const Viewer=dynamic(()=>import("../../../components/Viewer"),{ssr:false});
const TryOnViewer=dynamic(()=>import("../../../components/TryOnViewer"),{ssr:false});
const MeasurementPanel=dynamic(()=>import("../../../components/MeasurementPanel"),{ssr:false});
const API=process.env.NEXT_PUBLIC_API||"http://localhost:8000";

export default function Job(){
 const {id}=useParams<{id:string}>();
 const [s,setS]=useState<any>({status:"queued",progress:0});
 const [tab,setTab]=useState<"model"|"tryon">("model");
 const [url,setUrl]=useState("");
 useEffect(()=>{
  let timer:any;
  const poll=async()=>{
   try{
    const r=await fetch(API+"/api/jobs/"+id); const j=await r.json(); setS(j);
    if(j.status!=="done"&&j.status!=="failed")timer=setTimeout(poll,1200);
   }catch{}
  };
  poll(); return()=>clearTimeout(timer);
 },[id]);
 useEffect(()=>{
  if(s.glb_url)setUrl(s.glb_url.startsWith("http")?s.glb_url:API+s.glb_url);
 },[s.glb_url]);
 if(s.status==="failed")return <main style={{padding:32}}><h2>Generation failed</h2><pre>{s.error}</pre></main>;
 if(s.status!=="done")return <main style={{padding:32}}><h2>Generating…</h2><p>{Math.round((s.progress||0)*100)}% — {s.stage||s.status}</p></main>;
 const measurement=s.measurements||{};
 const onUpdated=(nextUrl:string,next:any)=>{setUrl(nextUrl.startsWith("http")?nextUrl:API+nextUrl);setS({...s,measurements:next});};
 return <main style={{maxWidth:1100,margin:"30px auto",padding:20}}>
  <h1>Generated glasses</h1>
  <div style={{display:"flex",gap:8,marginBottom:14}}>
   <button onClick={()=>setTab("model")} disabled={tab==="model"}>3D Model</button>
   <button onClick={()=>setTab("tryon")} disabled={tab==="tryon"}>Try On</button>
  </div>
  {tab==="model"?<Viewer url={url}/>:<TryOnViewer url={url}/>}
  <p><a href={url} download>Download GLB</a></p>
  <MeasurementPanel modelId={id} initial={measurement} onUpdated={onUpdated}/>
 </main>;
}
