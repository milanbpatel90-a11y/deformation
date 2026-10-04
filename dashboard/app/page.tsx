"use client";
import {useState} from "react";
import {useRouter} from "next/navigation";

const API=process.env.NEXT_PUBLIC_API||"http://localhost:8000";

export default function Home(){
 const [files,setFiles]=useState<File[]>([]); const [busy,setBusy]=useState(false); const router=useRouter();
 async function submit(){
  if(!files.length)return; setBusy(true); const fd=new FormData(); files.forEach(f=>fd.append("images",f));
  const r=await fetch(API+"/api/jobs",{method:"POST",body:fd}); if(!r.ok){setBusy(false);alert(await r.text());return}
  const j=await r.json(); router.push("/job/"+j.job_id);
 }
 return <main style={{maxWidth:760,margin:"60px auto",padding:24}}>
  <h1>Deformation VTO</h1><p>Upload 1–8 eyewear product images. Four to five views are recommended.</p>
  <input type="file" accept="image/*" multiple onChange={e=>setFiles(Array.from(e.target.files||[]))}/>
  <p>{files.length} image(s) selected</p>
  <button disabled={busy||!files.length} onClick={submit}>{busy?"Uploading…":"Generate 3D GLB"}</button>
 </main>
}
