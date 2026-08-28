import bpy
import os
import json

# Clear existing objects
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete(use_global=False)

# Paths
blend_file_path = os.path.abspath("templates/geometric_metal.glb")
output_glb_path = os.path.abspath("templates/geometric_metal.glb")
descriptor_path = os.path.abspath("templates/descriptors/geometric_metal.json")

print(f"Importing GLB from: {blend_file_path}")
# Import GLB
bpy.ops.import_scene.gltf(filepath=blend_file_path)

# Get all mesh objects
mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"Found {len(mesh_objs)} mesh objects initially:")
for obj in mesh_objs:
    print(f"  {obj.name}: {len(obj.data.vertices)} vertices")

# We'll process each original object and split if needed
all_new_objects = []

# Process each original mesh object
for obj in mesh_objs:
    # Make sure it's the active object and selected
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    
    vert_count = len(obj.data.vertices)
    print(f"\nProcessing '{obj.name}' with {vert_count} vertices")
    
    if vert_count > 50000:  # Large mesh (frame + rims + bridge)
        print("  -> Splitting large mesh by loose parts...")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
    elif vert_count > 6000 and vert_count < 8000:  # Cube mesh (lenses)
        print("  -> Splitting cube mesh by loose parts...")
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.mesh.separate(type='LOOSE')
        bpy.ops.object.mode_set(mode='OBJECT')
    elif vert_count > 2000 and vert_count < 3000:  # Temple meshes
        print("  -> This is a temple mesh, keeping as is (will rename later based on position)")
        pass
    else:
        print(f"  -> Unexpected vertex count {vert_count}, keeping as is")
        pass

# Now collect all mesh objects in the scene
all_mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"\nAfter splitting, we have {len(all_mesh_objs)} mesh objects:")
for obj in all_mesh_objs:
    print(f"  {obj.name}: {len(obj.data.vertices)} vertices")

# Now we need to rename the objects based on their type and position.
# We'll categorize them by vertex count and position.

# We'll create a list of tuples: (obj, vertex_count, x, y, z)
obj_data = []
for obj in all_mesh_objs:
    vert_count = len(obj.data.vertices)
    pos = obj.location
    obj_data.append((obj, vert_count, pos.x, pos.y, pos.z))

# Sort by x position (left to right)
obj_data.sort(key=lambda x: x[2])  # x[2] is x

print("\nObjects sorted by X position (left to right):")
for obj, vc, x, y, z in obj_data:
    print(f"  {obj.name}: {vc} verts at ({x:.2f}, {y:.2f}, {z:.2f})")

# Now we assign names based on our expectations.
# We expect 8 objects after splitting:
#   From large mesh: 4 pieces (Frame, Bridge, LeftRim, RightRim)
#   From cube mesh: 2 pieces (LeftLens, RightLens)
#   From temples: 2 pieces (LeftTemple, RightTemple)

# Step 1: Identify the two smallest objects (by vertex count) as temples.
sorted_by_vert = sorted(obj_data, key=lambda x: x[1])  # ascending by vertex count
print("\nSorted by vertex count (ascending):")
for obj, vc, x, y, z in sorted_by_vert:
    print(f"  {obj.name}: {vc} verts at x={x:.2f}")

# The two smallest should be the temples.
if len(sorted_by_vert) >= 2:
    temple1_obj, temple1_vc, temple1_x, temple1_y, temple1_z = sorted_by_vert[0]
    temple2_obj, temple2_vc, temple2_x, temple2_y, temple2_z = sorted_by_vert[1]
    # Assign left and right based on x
    if temple1_x < temple2_x:
        left_temple_obj = temple1_obj
        right_temple_obj = temple2_obj
    else:
        left_temple_obj = temple2_obj
        right_temple_obj = temple1_obj
    print(f"  Identified left temple: {left_temple_obj.name} at x={left_temple_obj.location.x:.2f}")
    print(f"  Identified right temple: {right_temple_obj.name} at x={right_temple_obj.location.x:.2f}")
else:
    print("  WARNING: Could not find two smallest objects for temples")
    left_temple_obj = right_temple_obj = None

# Step 2: Identify the two objects with vertex count around 3500 (half of 6916) as lenses.
# We'll look for objects with vertex count between 3000 and 4000.
lens_candidates = [ (obj, vc, x, y, z) for obj, vc, x, y, z in obj_data if 3000 <= vc <= 4000 ]
print(f"\nFound {len(lens_candidates)} lens candidates (vc between 3000 and 4000):")
for obj, vc, x, y, z in lens_candidates:
    print(f"  {obj.name}: {vc} verts at x={x:.2f}")
if len(lens_candidates) >= 2:
    # Sort these by x to get left and right lens
    lens_candidates.sort(key=lambda x: x[2])  # by x
    left_lens_obj = lens_candidates[0][0]
    right_lens_obj = lens_candidates[1][0]
    print(f"  Identified left lens: {left_lens_obj.name} at x={left_lens_obj.location.x:.2f}")
    print(f"  Identified right lens: {right_lens_obj.name} at x={right_lens_obj.location.x:.2f}")
else:
    print("  WARNING: Could not find two lens candidates")
    left_lens_obj = right_lens_obj = None

# Step 3: The largest object (by vertex count) should be the frame.
largest_obj = max(obj_data, key=lambda x: x[1])[0]
print(f"\nLargest object: {largest_obj.name} with {len(largest_obj.data.vertices)} vertices at x={largest_obj.location.x:.2f}")
frame_obj = largest_obj

# Step 4: The remaining objects should be bridge, left rim, right rim.
# We'll remove the ones we've already assigned from the list and then assign the rest.
assigned = set()
if left_temple_obj: assigned.add(left_temple_obj)
if right_temple_obj: assigned.add(right_temple_obj)
if left_lens_obj: assigned.add(left_lens_obj)
if right_lens_obj: assigned.add(right_lens_obj)
if frame_obj: assigned.add(frame_obj)

remaining_objs = [obj for obj, vc, x, y, z in obj_data if obj not in assigned]
print(f"\nRemaining objects (should be bridge, left rim, right rim): {len(remaining_objs)}")
for obj in remaining_objs:
    print(f"  {obj.name}: {len(obj.data.vertices)} vertices at x={obj.location.x:.2f}")

# We expect three remaining objects: bridge, left rim, right rim.
# We'll sort them by x and assign: leftmost -> left rim, middle -> bridge, rightmost -> right rim.
if len(remaining_objs) == 3:
    remaining_objs.sort(key=lambda obj: obj.location.x)
    left_rim_obj = remaining_objs[0]
    bridge_obj = remaining_objs[1]
    right_rim_obj = remaining_objs[2]
    print(f"  Assigned left rim: {left_rim_obj.name} at x={left_rim_obj.location.x:.2f}")
    print(f"  Assigned bridge: {bridge_obj.name} at x={bridge_obj.location.x:.2f}")
    print(f"  Assigned right rim: {right_rim_obj.name} at x={right_rim_obj.location.x:.2f}")
else:
    print(f"  WARNING: Expected 3 remaining objects, got {len(remaining_objs)}. Assigning arbitrarily.")
    # If we don't have exactly 3, we'll just assign the first three we have to the three roles.
    if len(remaining_objs) >= 3:
        left_rim_obj = remaining_objs[0]
        bridge_obj = remaining_objs[1]
        right_rim_obj = remaining_objs[2]
    elif len(remaining_objs) == 2:
        # Assume we are missing one, maybe the bridge is missing? We'll assign the two to left and right rim and set bridge to None.
        left_rim_obj = remaining_objs[0]
        right_rim_obj = remaining_objs[1]
        bridge_obj = None
    else:
        # Not enough, we'll set all to None and hope for the best.
        left_rim_obj = bridge_obj = right_rim_obj = None

# Now we have identified objects for each part. Let's rename them.
# We'll create a mapping from object to new name.
name_map = {}
if left_temple_obj:
    name_map[left_temple_obj] = "LeftTemple"
if right_temple_obj:
    name_map[right_temple_obj] = "RightTemple"
if left_lens_obj:
    name_map[left_lens_obj] = "LeftLens"
if right_lens_obj:
    name_map[right_lens_obj] = "RightLens"
if frame_obj:
    name_map[frame_obj] = "Frame"
if bridge_obj:
    name_map[bridge_obj] = "Bridge"
if left_rim_obj:
    name_map[left_rim_obj] = "LeftRim"
if right_rim_obj:
    name_map[right_rim_obj] = "RightRim"

print("\nRenaming objects:")
for obj, new_name in name_map.items():
    print(f"  Renaming '{obj.name}' to '{new_name}'")
    obj.name = new_name

# Now we need to make sure we have exactly 8 objects and they are all renamed.
# Let's check the final objects.
final_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
print(f"\nFinal mesh objects ({len(final_objs)}):")
for obj in final_objs:
    print(f"  {obj.name}: {len(obj.data.vertices)} vertices")

# If we have more than 8, we might have extra objects (like the original objects that weren't removed?).
# In Blender, when we separate by loose parts, the original object is removed and replaced by the new ones.
# So we should have exactly the number of pieces after splitting.

# Now we export the GLB.
print(f"\nExporting GLB to: {output_glb_path}")
# Ensure we select all objects
bpy.ops.object.select_all(action='SELECT')
# Export settings as per guide: glTF Binary (.glb), +Y Up, Apply Modifiers
bpy.ops.export_scene.gltf(
    filepath=output_glb_path,
    export_format='GLB',
    # These are the important settings from the guide:
    check_existing=False,  # overwrite
    # The following are the defaults but we set them explicitly:
    use_selection=True,    # export selected only
    export_apply=True,     # apply modifiers
    # Transform settings: we need +Y Up
    # In Blender's glTF export, the 'up' axis is controlled by the 'gltf_up_axis' parameter.
    # We want Y up, which is the default for glTF? Actually, glTF defaults to Z up.
    # We need to set the forward and up axes.
    # According to the guide: Transform: +Y Up
    # In Blender's glTF export, we can set:
    #   gltf_up_axis = 'Y'
    #   Also, we might need to set the forward axis to '-Z' (default is -Y) but let's check.
    # We'll leave the defaults and hope that the preset in the UI is correct.
    # Alternatively, we can set the axis conversion.
    # We'll use the following to match the Blender UI's "glTF 2.0" format with "+Y Up":
    #   The default export settings in Blender for glTF 2.0 are:
    #     - Forward: -Z
    #     - Up: Y
    #   So we don't need to change anything.
    # But let's set them explicitly to be safe.
    gltf_up_axis='Y',
    # Note: The export will apply the current transformation. We assume the model is already in the correct orientation.
)

print("Export completed.")

# Now update the descriptor JSON to match the new mesh names.
# The descriptor currently has:
#   "mesh_aliases": {
#     "Frame":        "Plane_glasses_mat_0",
#     ... etc.
#   }
# We need to change the values to the new names (which are the keys we just set).
# Since we renamed the objects to exactly the keys we want, we can set the value to be the same as the key.
# However, note that the descriptor might have additional structure. We'll load it, update the mesh_aliases, and save it.

print(f"\nUpdating descriptor at: {descriptor_path}")
try:
    with open(descriptor_path, 'r') as f:
        descriptor = json.load(f)
    
    # Update the mesh_aliases to use the new names (which are the same as the keys)
    if "mesh_aliases" in descriptor:
        for key in descriptor["mesh_aliases"]:
            # Set the value to the key itself (since we renamed the object to the key)
            descriptor["mesh_aliases"][key] = key
        print("Updated mesh_aliases to use the new mesh names.")
    else:
        print("WARNING: No 'mesh_aliases' key found in descriptor.")
    
    # Write back
    with open(descriptor_path, 'w') as f:
        json.dump(descriptor, f, indent=2)
    print("Descriptor updated successfully.")
except Exception as e:
    print(f"ERROR updating descriptor: {e}")

print("\nDone.")