"use client";

import {useState} from "react";

const API=process.env.NEXT_PUBLIC_API||"http://localhost:8000";

type Measurements={
 frame_width:number;bridge_width:number;lens_width:number;lens_height:number;
 temple_length:number;rim_thickness?:number;temple_curve_angle?:number;color?:string;
};

export default function MeasurementPanel({modelId,initial,onUpdated}:{modelId:string;initial:Measurements;onUpdated:(url:string,m:Measurements)=>void}){
 const [values,setValues]=useState<Measurements>(initial);
 const [busy,setBusy]=useState(false);
 const fields:(keyof Measurements)[]=["frame_width","bridge_width","lens_width","lens_height","temple_length","rim_thickness","temple_curve_angle"];
 async function save(){
  setBusy(true);
  try{
   const r=await fetch(API+"/api/models/"+modelId+"/adjust",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(values)});
   if(!r.ok)throw new Error(await r.text());
   const data=await r.json(); setValues(data.measurements); onUpdated(data.glb_url,data.measurements);
  }catch(e:any){alert(e?.message||"Adjustment failed");}
  finally{setBusy(false);}
 }
 return <section style={{marginTop:20,padding:18,border:"1px solid #ddd",borderRadius:12}}>
  <h2>Measurements</h2>
  <div style={{display:"grid",gridTemplateColumns:"repeat(2,minmax(0,1fr))",gap:12}}>
   {fields.map(field=><label key={field} style={{display:"grid",gap:4}}>
    <span>{field.replaceAll("_"," ")} (mm/°)</span>
    <input type="number" step="0.1" value={values[field]??""} onChange={e=>setValues({...values,[field]:Number(e.target.value)})}/>
   </label>)}
  </div>
  <button disabled={busy} onClick={save} style={{marginTop:14}}>{busy?"Re-deforming…":"Re-run deformation"}</button>
 </section>;
}
