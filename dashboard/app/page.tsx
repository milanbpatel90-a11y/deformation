"use client";

import {useState} from "react";
import {useRouter} from "next/navigation";

const API=process.env.NEXT_PUBLIC_API||"http://localhost:8000";

export default function Home(){
 const [files,setFiles]=useState<File[]>([]);
 const [video,setVideo]=useState<File|null>(null);
 const [busy,setBusy]=useState(false);
 const router=useRouter();
 async function submit(){
  if(!files.length&&!video)return;
  setBusy(true);
  const fd=new FormData();
  if(video) fd.append("video",video); else files.forEach(f=>fd.append("images",f));
  try{
   const r=await fetch(API+"/api/jobs",{method:"POST",body:fd});
   if(!r.ok)throw new Error(await r.text());
   const j=await r.json(); router.push("/job/"+j.job_id);
  }catch(e:any){alert(e?.message||"Upload failed");setBusy(false);}
 }
 return <main style={{maxWidth:760,margin:"60px auto",padding:24}}>
  <h1>Deformation VTO</h1>
  <p>Upload 4–5 product images or one short 360° orbit video.</p>
  <input type="file" accept="image/*" multiple disabled={!!video} onChange={e=>setFiles(Array.from(e.target.files||[]).slice(0,8))}/>
  <p>{files.length} image(s) selected</p>
  <div style={{margin:"20px 0"}}>or</div>
  <input type="file" accept="video/*" disabled={files.length>0} onChange={e=>setVideo(e.target.files?.[0]||null)}/>
  {video&&<p>Video: {video.name}</p>}
  <button disabled={busy||(!files.length&&!video)} onClick={submit}>{busy?"Uploading…":"Generate 3D GLB"}</button>
 </main>
}
