"""Diverse candidate ranking; prefer one strong frame per view before filling extras."""
def rank_frames(candidates,max_selected=24,per_view=3):
    ordered=sorted(candidates,key=lambda x:(x.get("quality_score",0)*0.55+x.get("detection_confidence",0)*0.3+x.get("view_confidence",0)*0.15),reverse=True)
    selected=[]; counts={}
    for item in ordered:
        view=item.get("view","unknown"); count=counts.get(view,0)
        if count>=per_view: continue
        selected.append(item); counts[view]=count+1
        if len(selected)>=max_selected: break
    return selected
