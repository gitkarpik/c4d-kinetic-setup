import os
import re
import json
import math
import time
from collections import OrderedDict
from itertools import accumulate

import c4d
from c4d.modules.mograph import FieldInput, FieldInfo, FieldOutput
from c4d.modules.tokensystem import FilenameConvertTokens

# Version 0.4.6 (Alpha)
# fixed initial jack positions on first ring



#todo:
#sync json export with Max - prepare basic animations

#add refresh on export settings update!
# add auto rotation field



fieldListExpand = op[c4d.ID_USERDATA, 2]
fieldListRot = op[c4d.ID_USERDATA, 5]
fieldListUp = op[c4d.ID_USERDATA, 4]

expandTime = op[c4d.ID_USERDATA, 27] * doc.GetFps()
expandCoolDown = op[c4d.ID_USERDATA, 7] * doc.GetFps()
expandCurve = op[c4d.ID_USERDATA, 6]


rotTime = op[c4d.ID_USERDATA, 8] * doc.GetFps()
rotCoolDown = op[c4d.ID_USERDATA, 28] * doc.GetFps()
rotCurve = op[c4d.ID_USERDATA, 21]

rotExpand = op[c4d.ID_USERDATA, 38] #thats a checkbox boolean, if crop to 30 or 45 degree

upTime = op[c4d.ID_USERDATA, 29] * doc.GetFps()
upCoolDown = op[c4d.ID_USERDATA, 30] * doc.GetFps()
upCurve = op[c4d.ID_USERDATA, 17]


displayScreens = op[c4d.ID_USERDATA, 16]
colorScreens = op[c4d.ID_USERDATA, 19]
uvScreens = op[c4d.ID_USERDATA, 20]

displayHelpers = op[c4d.ID_USERDATA, 25]
colorHelpers = op[c4d.ID_USERDATA, 26]


ringsCount = 30
hexagonsPerRing = 50
hexCount = hexagonsPerRing * ringsCount
hexPoints = 6

diameter_in = 501
diameter_out = 701
base_circRad = diameter_in / 2
ringHeight = 24.5
ringExpansion = 12.7
ringExpansion = 13.

color_ok = .5
color_error = 1.

# Animation parameters
expand_animation_length = expandTime
up_animation_length = upTime
rotation_animation_length = rotTime


def SetCurrentFrame(frame, doc):
    doc.SetTime(c4d.BaseTime(frame,doc.GetFps()))
    doc.ExecutePasses(None, True, True, True, 0)
    c4d.GeSyncMessage(c4d.EVMSG_TIMECHANGED)
    return

def create_sample_positions(ringsCount, hexagonsPerRing, ringHeight, base_circRad):
    samplePosList = []
    for ringIdx in range(ringsCount):
        for hexIdx in range(hexagonsPerRing):
            angle = (2 * math.pi / hexCount) * hexIdx * ringsCount
            angle += (ringIdx % 2) * 0.0628 * 5

            angle += 48.8 * math.pi / 180
            samplePosList.append(c4d.Vector(
                base_circRad * math.cos(angle),
                ringHeight*(ringIdx),
                base_circRad * math.sin(angle)
            ) * op.GetMg())
    return samplePosList

def prepare_field_inputs(samplePosList, ringsCount, hexCount, ringHeight):
    samplePosListExpand = [samplePosList[i+2] for i in range(0, hexCount, 5)]
    inputFieldExpand = c4d.modules.mograph.FieldInput(samplePosListExpand, hexCount//5)

    samplePosListUp = [c4d.Vector(0, i*ringHeight, 0) for i in range(0, ringsCount)]
    inputFieldUp = c4d.modules.mograph.FieldInput(samplePosListUp, ringsCount)

    inputFieldRot = c4d.modules.mograph.FieldInput(samplePosList, hexCount)

    return {
        "expand": inputFieldExpand,
        "up": inputFieldUp,
        "rotation": inputFieldRot
    }

def GetFieldData():
    # Record the start time to measure function execution time
    start_time = time.time()
    visibilityStatus = []

    # Append display and color settings from user data
    visibilityStatus.append(op[c4d.ID_USERDATA, 16])  # displayScreens
    visibilityStatus.append(op[c4d.ID_USERDATA, 19])  # colorScreens
    visibilityStatus.append(op[c4d.ID_USERDATA, 20])  # uvScreens
    visibilityStatus.append(op[c4d.ID_USERDATA, 25])  # displayHelpers
    visibilityStatus.append(op[c4d.ID_USERDATA, 26])  # colorHelpers

    op[c4d.ID_USERDATA, 16] = False
    op[c4d.ID_USERDATA, 19] = False
    op[c4d.ID_USERDATA, 20] = False
    op[c4d.ID_USERDATA, 25] = False
    op[c4d.ID_USERDATA, 26] = False


    doc = c4d.documents.GetActiveDocument()
    fps = doc.GetFps()
    startFrame = doc.GetLoopMinTime().GetFrame(fps)
    endFrame = doc.GetLoopMaxTime().GetFrame(fps)
    currentFrame = doc.GetTime().GetFrame(fps)

    samplePosList = create_sample_positions(ringsCount, hexagonsPerRing, ringHeight, base_circRad)

    field_inputs = prepare_field_inputs(samplePosList, ringsCount, hexCount, ringHeight)
    inputFieldExpand = field_inputs["expand"]
    inputFieldUp = field_inputs["up"]
    inputFieldRot = field_inputs["rotation"]

    field_configs = [
        {
            "name": "expand",
            "field_list": fieldListExpand,
            "input_field": inputFieldExpand,
            "count": hexCount//5,
            "prefix": "expand_",
            "animation_length": expand_animation_length,
            "cooldown_length": expandCoolDown,
            "default_value": 0
        },
        {
            "name": "up",
            "field_list": fieldListUp,
            "input_field": inputFieldUp,
            "count": ringsCount,
            "prefix": "up_",
            "animation_length": up_animation_length,
            "cooldown_length": upCoolDown,
            "default_value": 0
        },
        {
            "name": "rotation",
            "field_list": fieldListRot,
            "input_field": inputFieldRot,
            "count": hexCount,
            "prefix": "rotation_",
            "animation_length": rotation_animation_length,
            "cooldown_length": rotCoolDown,
            "default_value": 0.5
        }
    ]

    field_data = {}
    prev_values = {}
    prev_frame_values = {}
    active_animations = {}

    for config in field_configs:
        prefix = config["prefix"]
        count = config["count"]
        anim_length = config["animation_length"]

        for i in range(count):
            field_key = f"{prefix}{i}"
            field_data[field_key] = {"operations": []}

        prev_values[config["name"]] = [config["default_value"]] * count
        prev_frame_values[config["name"]] = [config["default_value"]] * count
        active_animations[config["name"]] = [None] * count

    # OPTIMIZATION: Pre-sample all frames at once to avoid repeated sampling
    sample_time = 0
    all_samples = {}

    # Initialize the sample storage for each field
    for config in field_configs:
        name = config["name"]
        all_samples[name] = []

    # Sample all frames in one go
    for frame in range(startFrame, endFrame + 1):
        sample_start_time = time.time()
        SetCurrentFrame(frame, doc)

        frame_samples = {}
        for config in field_configs:
            name = config["name"]
            field_list = config["field_list"]
            input_field = config["input_field"]
            prefix = config["prefix"]
            count = config["count"]

            samp = field_list.SampleListSimple(op, input_field, c4d.FIELDSAMPLE_FLAG_VALUE)
            # Store the rounded values to avoid floating point precision issues
            frame_samples[name] = [round(samp._value[i]*100)/100 for i in range(len(samp._value))]
            #here add snapping to 0., 0.333, 0.666, 1. for lift field
            if name == "up":
                for i in range(len(samp._value)):
                    val = samp._value[i]
                    # Find closest snap point
                    snap_points = [0.0, 0.333, 0.666, 1.0]
                    closest = min(snap_points, key=lambda x: abs(x - val))
                    frame_samples[name][i] = closest

            if frame == startFrame:
                prev_frame_values[name] = frame_samples[name].copy()
                prev_values[name] = frame_samples[name].copy()
                for i in range(count):
                    field_key = f"{prefix}{i}"
                    field_data[field_key]["operations"].append({"initial_value": frame_samples[name][i]})


        frame_samples["up"][0] = field_data["up_0"]["operations"][0]["initial_value"]
        all_samples[frame] = frame_samples

        sample_time += time.time() - sample_start_time



    print(f"Field sampling completed in {sample_time:.2f} seconds")

    # Process the pre-sampled data to detect changes
    processing_start_time = time.time()
    for frame in range(startFrame, endFrame + 1):
        frame_samples = all_samples[frame]

        for config in field_configs:
            name = config["name"]
            prefix = config["prefix"]
            anim_length = config["animation_length"]
            cooldown_length = config["cooldown_length"]

            samples = frame_samples[name]

            for i in range(len(samples)):
                current_value = samples[i]
                prev_value = prev_values[name][i]
                if name == "rotation":
                    # Calculate corresponding up index - rotation prefix number divided by hexagonsPerRing
                    rot_index = int(i)
                    up_index = rot_index // hexagonsPerRing
                    field_key_up = f"up_{up_index}"

                    # Find the last UP command for this index
                    up_value = field_data[field_key_up]["operations"][0]["initial_value"]
                    #print(field_data[field_key_up]["operations"])

                    if field_key_up in field_data:
                        for operation in reversed(field_data[field_key_up]["operations"]):
                            #print(f"Operation: {operation} field_key_up: {field_key_up}")
                            if "dest_value" in operation and operation["frame"] <= frame:
                                up_value = operation["dest_value"]
                                #print(f"Up value: {up_value}")
                                break

                    field_key_up_2 = f"up_{max(up_index - 1, 1)}"
                    up_value_2 = field_data[field_key_up_2]["operations"][0]["initial_value"]
                    if field_key_up_2 in field_data:
                        for operation in reversed(field_data[field_key_up_2]["operations"]):
                            if "dest_value" in operation and operation["frame"] <= frame:
                                up_value_2 = operation["dest_value"]
                                break
                    min_up_value = min(up_value, up_value_2)
                    min_up_value = up_value

                    # Check expansion value for this row
                    expand_index = rot_index // 5
                    field_key_expand = f"expand_{expand_index}"
                    expand_value = field_data[field_key_expand]["operations"][0]["initial_value"]

                    if field_key_expand in field_data:
                        for operation in reversed(field_data[field_key_expand]["operations"]):
                            if "dest_value" in operation and operation["frame"] <= frame:
                                expand_value = operation["dest_value"]
                                break

                    if min_up_value is not None:
                        if min_up_value < 0.333:
                            current_val_clamped = 0.5  # Fixed at 0 degree
                        elif min_up_value < 0.666:
                            current_val_clamped = min(max(current_value, 0.4), 0.6)  # +- 10 degree
                        else:
                            current_val_clamped = min(max(current_value, 0.17), 0.82)  # +- 30 degree
                            if expand_value > 0.2 and rotExpand == 1:
                                current_val_clamped = current_value  # No clamp (45 deg)


                    current_value = current_val_clamped
                    #prev_value = prev_val_clamped
                    #current_value = 0.5
                        # Update the samples with clamped value
                        #frame_samples[name][i] = current_value

                if abs(current_value - prev_frame_values[name][i]) > 0.0101:

                    change_magnitude = abs(current_value - prev_values[name][i])
                    animation_length = max(int(change_magnitude * anim_length), 1)

                    is_valid = True
                    if active_animations[name][i] is not None and frame < active_animations[name][i]:
                        is_valid = False

                    field_key = f"{prefix}{i}"
                    field_data[field_key]["operations"].append({
                        "frame": frame,
                        "start_value": prev_value,
                        "dest_value": current_value,
                        "animation_length": animation_length,
                        "valid": is_valid
                    })

                    if is_valid:
                        active_animations[name][i] = frame + animation_length + cooldown_length
                        prev_values[name][i] = current_value
                    prev_frame_values[name][i] = current_value

    SetCurrentFrame(currentFrame, doc)

    op[c4d.ID_USERDATA, 16] = visibilityStatus[0]
    op[c4d.ID_USERDATA, 19] = visibilityStatus[1]
    op[c4d.ID_USERDATA, 20] = visibilityStatus[2]
    op[c4d.ID_USERDATA, 25] = visibilityStatus[3]
    op[c4d.ID_USERDATA, 26] = visibilityStatus[4]

    # Print time difference
    end_time = time.time()
    time_diff = end_time - processing_start_time
    print(f"Field data processing completed in {time_diff:.2f} seconds")
    #print(field_data)
    return field_data

def SaveJsonData(doc, filtered_data):
    # Get the path of the current C4D project file
    project_path = doc.GetDocumentPath()
    project_name = doc.GetDocumentName()
    # Create the output file path in the same directory
    output_filename = "" + project_name[:-4] + ".json"


    renderData = doc.GetActiveRenderData()
    renderSettings = renderData.GetDataInstance()
    frame = doc.GetTime().GetFrame(doc.GetFps())
    rpd = {'_doc': doc, '_rData': renderData, '_rBc': renderSettings, '_frame': frame}
    filePath = os.path.normpath(os.path.join(doc.GetDocumentPath(), FilenameConvertTokens(op[c4d.ID_USERDATA,32], rpd) + '.json'))
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.makedirs(directory)


    output_path = filePath

    # Reorganize data into the new format
    fps = doc.GetFps()
    total_frames = doc.GetLoopMaxTime().GetFrame(fps) - doc.GetLoopMinTime().GetFrame(fps) + 1

    # Get current timestamp
    current_time = time.strftime("%Y-%m-%d %H:%M:%S")

    # Create the new data structure
    new_data = {
        "name": project_name[:-4],
        "type": "kinematic_preset",
        "data": {},
        "info": {
            "created_at": current_time,
            "version": "1",
            "export_range": {
                "start_frame": doc.GetLoopMinTime().GetFrame(fps),
                "end_frame": doc.GetLoopMaxTime().GetFrame(fps)
            },
            "total_frames": total_frames,
            "part" : 1,
            "total_parts": 1,
            "fps": fps
        }
    }

    # Group operations by row and type
    for key, data in filtered_data.items():
        if key.startswith("expand_"):
            type_name = "pusher"
            index = int(key.split("_")[1])
            motor_id = "id_" + str(index%10 + 1)
            row_index = (index//10) + 1 # Calculate which row this belongs to
        elif key.startswith("up_"):
            type_name = "jack"
            index = int(key.split("_")[1])
            motor_id = "id_1" #+ str(index + 1)
            row_index = index + 1  # Row index is directly in the key
        elif key.startswith("rotation_"):
            type_name = "tilt"
            index = int(key.split("_")[1])
            motor_id = "id_" + str(index%hexagonsPerRing + 1)
            row_index = index // hexagonsPerRing + 1  # Calculate which row this belongs to
        else:
            continue

        row_key = f"row_{row_index}"

        # Create row if it doesn't exist
        if row_key not in new_data["data"]:
            new_data["data"][row_key] = {}

        # Create type if it doesn't exist
        if type_name not in new_data["data"][row_key]:
            new_data["data"][row_key][type_name] = {}



        # Add operations
        for i, operation in enumerate(data["operations"]):
            if "initial_value" in operation:
                continue
            if motor_id not in new_data["data"][row_key][type_name]:
                new_data["data"][row_key][type_name][motor_id] = []
            op_id = f"op_{i+1}"
            if type_name == "jack":
                operation["start_value"] = round(operation["start_value"] * 3)
                operation["dest_value"] = round(operation["dest_value"] * 3)
            if type_name == "tilt":
                operation["start_value"] = round(operation["start_value"] - 0.5, 2)
                operation["dest_value"] = round(operation["dest_value"] - 0.5, 2)

            operation["frame"] = operation["frame"] - doc.GetLoopMinTime().GetFrame(fps)

            start_val = operation["start_value"]
            dest_val = operation["dest_value"]
            length_val = operation["animation_length"]
            frame = operation["frame"]
            new_op = {
                "frame": frame,
                "start": start_val,
                "dest": dest_val,
                "length": length_val,
            }


            # Convert animation_length from frames to seconds
            #if "animation_length" in operation:
            #    operation["animation_length"] = round(operation["animation_length"] / fps, 2)

            # Remove the "valid" field from the operation
            if "valid" in operation:
                del operation["valid"]

            #new_data["data"][row_key][type_name][motor_id].append(operation)
            new_data["data"][row_key][type_name][motor_id].append(new_op)

    # Save the JSON data to file
    with open(output_path, 'w') as f:
        json.dump(new_data, f, indent=4)

    print(f"Field data saved to: {output_path}")


def SaveFieldDataToJson(field_data):
    doc = c4d.documents.GetActiveDocument()

    # Filter out invalid operations
    filtered_data = {}
    # Flag to control filtering (can be changed to False to include invalid operations)
    filter_invalid = False

    # Track frames with invalid operations and count them
    invalid_frames = {}

    for key, data in field_data.items():
        # Check for invalid operations and collect their frames with counts
        for op in data["operations"]:
            if not op.get("valid", True):
                frame = op["frame"]
                if frame in invalid_frames:
                    invalid_frames[frame] += 1
                else:
                    invalid_frames[frame] = 1

        if filter_invalid:
            filtered_operations = [op for op in data["operations"] if op.get("valid", True)]
        else:
            filtered_operations = data["operations"]
        filtered_data[key] = {"operations": filtered_operations}


    if invalid_frames:
        # Create a readable message listing all frames with invalid operations
        invalid_frames_message = "Invalid operations found at frames:\n"
        for frame, count in sorted(invalid_frames.items()):
            if len(invalid_frames_message.split('\n')) < 15:
                invalid_frames_message += f"Frame {frame}: {count}\n"
            else:
                invalid_frames_message += "..."

        invalid_frames_message += "\nExport data anyway?"
        #print(invalid_frames_message)
        dialog = c4d.gui.MessageDialog(invalid_frames_message, c4d.GEMB_OKCANCEL)
        if dialog == c4d.GEMB_R_OK:
            SaveJsonData(doc, filtered_data)
    else:
        print("No invalid operations found")
        SaveJsonData(doc, filtered_data)
    return

def set_error_color_keyframes(track, frame, fps, color_error, color_ok):
    """
    Sets error color keyframes for a given track at a specific frame.

    Args:
        track: The color track to add keyframes to
        frame: The frame where the error occurs
        fps: Frames per second of the document
        color_error: The color value to use for error indication
        color_ok: The color value to use for normal state
    """
    curve = track.GetCurve()
    start_time = c4d.BaseTime(frame, fps)
    end_time = c4d.BaseTime(frame + 1, fps)

    key_start = curve.AddKey(start_time)["key"]
    track.FillKey(doc, track.GetObject(), key_start)
    key_start.SetValue(curve, color_error)
    key_start.SetInterpolation(curve, c4d.CINTERPOLATION_STEP)

    key_end = curve.AddKey(end_time)["key"]
    track.FillKey(doc, track.GetObject(), key_end)
    key_end.SetValue(curve, color_ok)
    key_end.SetInterpolation(curve, c4d.CINTERPOLATION_STEP)

def add_keyframes_to_track(track, obj, start_frame, end_frame, start_value, end_value, remap_curve):
    """
    Adds keyframes to a track with specified start and end values.

    Args:
        track: The track to add keyframes to
        obj: The object the track belongs to
        start_time: The time for the start keyframe
        end_time: The time for the end keyframe
        start_value: The value for the start keyframe
        end_value: The value for the end keyframe
    """
    doc = c4d.documents.GetActiveDocument()
    fps = doc.GetFps()
    fps_mult = 100.

    knots = remap_curve.GetKnots()
    if knots[0]['interpol'] == c4d.CustomSplineKnotInterpolationBezier:
        tangentRight = knots[0]['vTangentRight']
    else:
        tangentRight = c4d.Vector(0, 0, 0)
    if knots[1]['interpol'] == c4d.CustomSplineKnotInterpolationBezier:
        tangentLeft = knots[1]['vTangentLeft']
    else:
        tangentLeft = c4d.Vector(0, 0, 0)

    anim_length = end_frame - start_frame

    start_time = c4d.BaseTime(start_frame, fps)
    end_time = c4d.BaseTime(end_frame, fps)

    curve = track.GetCurve()

    # Start keyframe
    key_start = curve.AddKey(start_time)["key"]
    track.FillKey(doc, obj, key_start)
    key_start.SetValue(curve, start_value)
    key_start.SetInterpolation(curve, c4d.CINTERPOLATION_SPLINE)

    key_start.SetTimeRight(curve, c4d.BaseTime(anim_length*tangentRight[0]*fps_mult, fps*fps_mult))
    key_start.SetValueRight(curve, c4d.utils.RangeMap(tangentRight[1], 0, 1, 0, end_value-start_value, False))


    # End keyframe
    key_end = curve.AddKey(end_time)["key"]
    track.FillKey(doc, obj, key_end)
    key_end.SetValue(curve, end_value)
    key_end.SetInterpolation(curve, c4d.CINTERPOLATION_SPLINE)

    key_end.SetTimeLeft(curve, c4d.BaseTime(anim_length*tangentLeft[0]*fps_mult, fps*fps_mult))
    key_end.SetValueLeft(curve, c4d.utils.RangeMap(tangentLeft[1], 0, 1, 0, start_value-end_value, False))

    return key_start, key_end

def Bake(field_data):
    start_time_bake = time.time()
    doc = c4d.documents.GetActiveDocument()
    currentTime = doc.GetTime()
    currentFrame = doc.GetTime().GetFrame(doc.GetFps())

    fps = doc.GetFps()
    startFrame = doc.GetLoopMinTime().GetFrame(fps)
    endFrame = doc.GetLoopMaxTime().GetFrame(fps)

    position_descriptors = {
        'X': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                        c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)),
        'Y': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                        c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)),
        'Z': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_POSITION, c4d.DTYPE_VECTOR, 0),
                        c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0))
    }

    rotation_descriptors = {
        'H': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0),
                        c4d.DescLevel(c4d.VECTOR_X, c4d.DTYPE_REAL, 0)),
        'P': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0),
                        c4d.DescLevel(c4d.VECTOR_Y, c4d.DTYPE_REAL, 0)),
        'B': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_REL_ROTATION, c4d.DTYPE_VECTOR, 0),
                        c4d.DescLevel(c4d.VECTOR_Z, c4d.DTYPE_REAL, 0))
    }

    color_descriptors = {
        'R': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_COLOR, c4d.DTYPE_COLOR, 0),
                        c4d.DescLevel(c4d.COLOR_R, c4d.DTYPE_REAL, 0)),
        'G': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_COLOR, c4d.DTYPE_COLOR, 0),
                        c4d.DescLevel(c4d.COLOR_G, c4d.DTYPE_REAL, 0)),
        'B': c4d.DescID(c4d.DescLevel(c4d.ID_BASEOBJECT_COLOR, c4d.DTYPE_COLOR, 0),
                        c4d.DescLevel(c4d.COLOR_B, c4d.DTYPE_REAL, 0))
    }

    settings = c4d.BaseContainer()
    baked_gen = doc.SearchObject(op.GetName() + "_Baked")
    if baked_gen:
        baked_gen.Remove()
    SetCurrentFrame(startFrame, doc)
    # Create structure to remember visibility status
    visibilityStatus = []

    # Append display and color settings from user data
    visibilityStatus.append(op[c4d.ID_USERDATA, 16])  # displayScreens
    visibilityStatus.append(op[c4d.ID_USERDATA, 19])  # colorScreens
    visibilityStatus.append(op[c4d.ID_USERDATA, 20])  # uvScreens
    visibilityStatus.append(op[c4d.ID_USERDATA, 25])  # displayHelpers
    visibilityStatus.append(op[c4d.ID_USERDATA, 26])  # colorHelpers


    op[c4d.ID_USERDATA, 16] = True
    op[c4d.ID_USERDATA, 19] = False
    op[c4d.ID_USERDATA, 20] = True
    op[c4d.ID_USERDATA, 25] = False
    op[c4d.ID_USERDATA, 26] = False

    baked_gen = c4d.utils.SendModelingCommand(
                command = c4d.MCOMMAND_CURRENTSTATETOOBJECT,
                list = [op],
                mode=c4d.MODELINGCOMMANDMODE_ALL,
                bc=settings,
                doc = doc)[0]
    baked_gen.SetName(op.GetName() + "_Baked")
    doc.InsertObject(baked_gen)
    # Remove material tag from baked generator
    mat_tag = baked_gen.GetTag(c4d.Ttexture)
    if mat_tag and uvScreens == False:
        mat_tag.Remove()

    
    visNull = baked_gen.GetDown().GetNext()
    if(visNull):
        visNull.Remove()
    # Create a list of all screen objects
    all_screens = []
    row_nulls = []
    # Function to recursively find all row nulls and screen objects in reverse order
    def collect_objects(parent, row_list, screen_list):
        # Get all children and reverse the order
        children = list(parent.GetChildren())
        children.reverse()

        for child in children:
            if child.GetName().startswith("row_"):
                row_list.append(child)
                # Recursively process children of this row
                collect_objects(child, row_list, screen_list)
            elif child.GetName() == "screens":
                # Add all screen objects under this screens null in reverse order
                screens = list(child.GetChildren())
                for screen in screens:
                    screen_list.append(screen)

    # Start the recursive collection from the baked generator
    collect_objects(baked_gen.GetDown(), row_nulls, all_screens)

    # Now all_screens contains all the screen objects from all rows in reverse order
    #print(f"Found {len(all_screens)} screen objects")
    #print(f"Found {len(row_nulls)} row nulls")

    # Create tracks for each child object
    for baked_scr in all_screens:
        # Create position tracks
        for axis, desc in position_descriptors.items():
            if(axis == 'Y'):
                continue
            track = baked_scr.FindCTrack(desc)
            if not track:
                track = c4d.CTrack(baked_scr, desc)
                baked_scr.InsertTrackSorted(track)

        # Create rotation tracks
        for axis, desc in rotation_descriptors.items():
            track = baked_scr.FindCTrack(desc)
            if not track:
                track = c4d.CTrack(baked_scr, desc)
                baked_scr.InsertTrackSorted(track)

        # Create color tracks
        for color, desc in color_descriptors.items():
            track = baked_scr.FindCTrack(desc)
            if not track:
                track = c4d.CTrack(baked_scr, desc)
                baked_scr.InsertTrackSorted(track)

    for row_null in row_nulls:
        # Create position tracks
        for axis, desc in position_descriptors.items():
            if(axis == 'Y'):
                track = row_null.FindCTrack(desc)
                if not track:
                    track = c4d.CTrack(row_null, desc)
                    row_null.InsertTrackSorted(track)

    # Set initial color to black (0,0,0) at frame 0
    for baked_scr in all_screens:
        # Set keyframes for each color component (R, G, B) at frame 0
        for color_component, desc in color_descriptors.items():
            track = baked_scr.FindCTrack(desc)
            if track:
                curve = track.GetCurve()
                start_time = c4d.BaseTime(0, fps)  # Frame 0

                # Add key at frame 0
                key = curve.AddKey(start_time)["key"]
                track.FillKey(doc, baked_scr, key)
                key.SetValue(curve, color_ok)  # Set to 0 (black)
                key.SetInterpolation(curve, c4d.CINTERPOLATION_STEP)

   # Map field data keys to object indices
    expand_objects = {}
    up_objects = {}
    rotation_objects = {}

    # Process field data and create sets for keyframes
    for key, data in field_data.items():
        # Filter out the initial value entry
        operations = [op for op in data["operations"] if "initial_value" not in op]
        filtered_data = {"operations": operations}

        if key.startswith("expand_"):
            index = int(key.split("_")[1])
            expand_objects[index] = filtered_data
        elif key.startswith("up_"):
            index = int(key.split("_")[1])
            up_objects[index] = filtered_data
        elif key.startswith("rotation_"):
            index = int(key.split("_")[1])
            rotation_objects[index] = filtered_data



    # Process each hexagon
    for i, baked_scr in enumerate(all_screens):
        baked_scr[c4d.ID_BASEOBJECT_QUATERNION_ROTATION_INTERPOLATION] = 1

        ringIdx = i // hexagonsPerRing
        hexIdx = i % hexagonsPerRing

        # Get initial position and rotation
        initial_pos = baked_scr[c4d.ID_BASEOBJECT_REL_POSITION]
        initial_rot = baked_scr[c4d.ID_BASEOBJECT_REL_ROTATION]

        # Process rotation animations
        if i in rotation_objects:
            for op_data in rotation_objects[i]["operations"]:
                frame = op_data["frame"]
                if not op_data.get("valid", True):
                    if frame > 0:
                        track = baked_scr.FindCTrack(color_descriptors['G'])
                        set_error_color_keyframes(track, frame, fps, color_error, color_ok)
                    continue

                field_start_value = op_data["start_value"]
                field_dest_value = op_data["dest_value"]
                anim_length = max(op_data["animation_length"], 1)

                # Calculate rotation value based on position
                rotXAngle = c4d.utils.RangeMap(field_start_value, 0., 1., -45., 45., False) * math.pi / 180

                # Calculate base Y rotation based on hexagon position
                angle = (2 * math.pi / hexCount) * hexIdx * ringsCount
                angle += (ringIdx % 2) * 0.0628 * 5  # Offset for alternating rings

                angle += 48.8 * math.pi / 180
                baseYAngle = angle + math.pi/2.

                matrix = c4d.Matrix()
                matrix_y = c4d.utils.MatrixRotY(baseYAngle)
                matrix_x = c4d.utils.MatrixRotX(rotXAngle)
                initial_hpb = c4d.utils.MatrixToHPB(matrix * matrix_y * matrix_x)


                rotXAngle = c4d.utils.RangeMap(field_dest_value, 0., 1., -45., 45., False) * math.pi / 180

                matrix = c4d.Matrix()
                matrix_y = c4d.utils.MatrixRotY(baseYAngle)
                matrix_x = c4d.utils.MatrixRotX(rotXAngle)
                hpb = c4d.utils.MatrixToHPB(matrix * matrix_y * matrix_x)


                start_time = frame
                end_time = frame + anim_length

                # Create rotation keyframes for all axes
                for axis_idx, axis in enumerate(['H', 'P', 'B']):
                    axis_map = {'H': 'X', 'P': 'Y', 'B': 'Z'}
                    track = baked_scr.FindCTrack(rotation_descriptors[axis])
                    add_keyframes_to_track(track, baked_scr, start_time, end_time, initial_hpb[axis_idx], hpb[axis_idx], rotCurve)

        # Process expand animations (affects X and Z position)
        expand_index = i // 5
        if expand_index in expand_objects:
            for op_data in expand_objects[expand_index]["operations"]:
                frame = op_data["frame"]
                if not op_data.get("valid", True):
                    if frame > 0:
                        track = baked_scr.FindCTrack(color_descriptors['R'])
                        set_error_color_keyframes(track, frame, fps, color_error, color_ok)
                    continue

                field_start_value = op_data["start_value"]
                field_dest_value = op_data["dest_value"]
                anim_length = op_data["animation_length"]

                # Calculate radius based on position
                current_circRad = c4d.utils.RangeMap(field_start_value, 0., 1., diameter_in, diameter_out, True) / 2.

                # Calculate X and Z position based on angle and radius
                angle = (2 * math.pi / hexCount) * hexIdx * ringsCount
                angle += (ringIdx % 2) * 0.0628 * 5  # Offset for alternating rings

                angle += 48.8 * math.pi / 180
                pos_x_start = current_circRad * math.cos(angle)
                pos_z_start = current_circRad * math.sin(angle)

                current_circRad = c4d.utils.RangeMap(field_dest_value, 0., 1., diameter_in, diameter_out, True) / 2.

                pos_x_dest = current_circRad * math.cos(angle)
                pos_z_dest = current_circRad * math.sin(angle)


                # Set start keyframe
                start_time = frame

                # Set end keyframe
                end_time = frame + anim_length

                # Create X position keyframe
                track_x = baked_scr.FindCTrack(position_descriptors['X'])
                add_keyframes_to_track(track_x, baked_scr, start_time, end_time, pos_x_start, pos_x_dest, expandCurve)

                # Create Z position keyframe
                track_z = baked_scr.FindCTrack(position_descriptors['Z'])
                add_keyframes_to_track(track_z, baked_scr, start_time, end_time, pos_z_start, pos_z_dest, expandCurve)

        # Process up animations (affects Y position)
        if ringIdx in up_objects:
            for op_data in up_objects[ringIdx]["operations"]:
                frame = op_data["frame"]
                if not op_data.get("valid", True):
                    if frame > 0:
                        track = baked_scr.FindCTrack(color_descriptors['B'])
                        set_error_color_keyframes(track, frame, fps, color_error, color_ok)
                    continue


                field_start_value = op_data["start_value"]
                field_dest_value = op_data["dest_value"]
                anim_length = op_data["animation_length"]

                # Calculate height based on position
                height_start = c4d.utils.RangeMap(field_start_value, 0., 1., 0, ringExpansion, False)
                height_start = ringHeight + height_start

                height_dest = c4d.utils.RangeMap(field_dest_value, 0., 1., 0, ringExpansion, False)
                height_dest = ringHeight + height_dest


                start_time = frame
                end_time = frame + anim_length

                track_y = row_nulls[ringIdx].FindCTrack(position_descriptors['Y'])
                add_keyframes_to_track(track_y, row_nulls[ringIdx], start_time, end_time, height_start, height_dest, upCurve)

    # Restore original time
    doc.SetTime(currentTime)

    op[c4d.ID_USERDATA, 16] = visibilityStatus[0]
    op[c4d.ID_USERDATA, 19] = visibilityStatus[1]
    op[c4d.ID_USERDATA, 20] = visibilityStatus[2]
    op[c4d.ID_USERDATA, 25] = visibilityStatus[3]
    op[c4d.ID_USERDATA, 26] = visibilityStatus[4]

    doc.ExecutePasses(None, True, True, True, 0)
    c4d.GeSyncMessage(c4d.EVMSG_TIMECHANGED)

    # Offset the baked generator for visibility
    baked_gen[c4d.ID_BASEOBJECT_REL_POSITION, c4d.VECTOR_Z] = 870.

    # Print time difference
    end_time_bake = time.time()
    time_diff_bake = end_time_bake - start_time_bake
    print(f"Bake completed in {time_diff_bake:.2f} seconds")

    return baked_gen

def message(id, data):
    if(id==c4d.MSG_DESCRIPTION_POSTSETPARAMETER):
        flId = eval(str(data['descid']))[1][0]
        if flId in [2, 4, 5, 20, 16, 24, 25, 26, 27,7,6,8,28,21,29,30,17,38]:
            c4d.CallButton(op, c4d.OPYTHON_MAKEDIRTY)
            #fieldListExpand = op[c4d.ID_USERDATA, 2]
            #fieldListUp = op[c4d.ID_USERDATA, 4]
            #fieldListRot = op[c4d.ID_USERDATA, 5]
    if id == c4d.MSG_DESCRIPTION_COMMAND:
        buttID = str(data['id']).split("), (")[1].split(", ")[0].strip("()")
        if(buttID=='9'):
            field_data = GetFieldData()
            Bake(field_data)
        elif(buttID=='10'):
            field_data = GetFieldData()
            SaveFieldDataToJson(field_data)
        elif(buttID=='23'):
            field_data = GetFieldData()
            SaveImageSequence(field_data)

def DrawSquare(texture, x, y, w, h, color):
    for i in range(x, x+w):
        for j in range(y-h, y):
            texture.SetPixel(i, j, int(color[0]), int(color[1]), int(color[2]))

def DrawPoly(texture, points, color):
    # Find bounding box of polygon
    min_x = min(p[0] for p in points)
    max_x = max(p[0] for p in points)
    min_y = min(p[1] for p in points)
    max_y = max(p[1] for p in points)

    # For each pixel in bounding box
    for x in range(int(min_x), int(max_x)+1):
        for y in range(int(min_y), int(max_y)+1):
            inside = False
            j = len(points) - 1

            # Ray casting algorithm to determine if point is inside polygon
            for i in range(len(points)):
                if ((points[i][1] > y) != (points[j][1] > y) and
                    x < (points[j][0] - points[i][0]) * (y - points[i][1]) /
                    (points[j][1] - points[i][1]) + points[i][0]):
                    inside = not inside
                j = i

            if inside:
                texture.SetPixel(x, y, int(color[0]), int(color[1]), int(color[2]))

def SaveImageSequence(field_data):
    #https://www.youtube.com/watch?v=R6_GQw-4tJY&t=310s
    #https://developers.maxon.net/docs/py/2024_0_0a/modules/c4d.modules/tokensystem/index.html
    #original resolution 2500x1150
    res = [2500, 1150]
    res = [res[0]//2, res[1]//2]

    doc = c4d.documents.GetActiveDocument()
    renderData = doc.GetActiveRenderData()
    renderSettings = renderData.GetDataInstance()
    frame = doc.GetTime().GetFrame(doc.GetFps())  # Current frame, or use 1 if you want frame 1

    fps = doc.GetFps()
    startFrame = doc.GetLoopMinTime().GetFrame(fps)
    endFrame = doc.GetLoopMaxTime().GetFrame(fps)

    #plan for the function:
    #for all frames in range - run a loop
    #then for each field data (expand, up, rotation) - find closest keyframe, from the left and from the right
    #interpolate them with corresponding curves from user data
    #write onto pixel map, using the uv coordinates of the hexagon

    # Map field data keys to object indices
    expand_objects = {}
    up_objects = {}
    rotation_objects = {}
    #print(field_data)
    # Process field data and create sets for keyframes
    for key, data in field_data.items():
        # Filter out the initial value entry
        operations = [op for op in data["operations"] if "initial_value" not in op]
        initial_value = [op for op in data["operations"] if "initial_value" in op]

        filtered_data = {"operations": operations, "initial_value": initial_value[0]["initial_value"]}

        if key.startswith("expand_"):
            index = int(key.split("_")[1])
            expand_objects[index] = filtered_data
        elif key.startswith("up_"):
            index = int(key.split("_")[1])
            up_objects[index] = filtered_data
        elif key.startswith("rotation_"):
            index = int(key.split("_")[1])
            rotation_objects[index] = filtered_data
    print(expand_objects)

    # Create directory if it doesn't exist
    rpd = {'_doc': doc, '_rData': renderData, '_rBc': renderSettings, '_frame': startFrame}
    filePath = os.path.normpath(os.path.join(doc.GetDocumentPath(), FilenameConvertTokens(op[c4d.ID_USERDATA,24], rpd) + '.png'))
    directory = os.path.dirname(filePath)
    if not os.path.exists(directory):
        os.makedirs(directory)


    for fr in range(startFrame, endFrame):
        rpd = {'_doc': doc, '_rData': renderData, '_rBc': renderSettings, '_frame': fr}
        filePath = os.path.normpath(os.path.join(doc.GetDocumentPath(), FilenameConvertTokens(op[c4d.ID_USERDATA,24], rpd) + '.png'))

        texture = c4d.bitmaps.BaseBitmap()
        texture.Init(res[0], res[1])
        for i in range(hexCount):
            ringIdx = i // hexagonsPerRing
            hexIdx = i % hexagonsPerRing
            # Find closest expand operations for current frame
            expandId = i//5
                # Get all operation frames for this hexagon
            operation_frames = [op["frame"] for op in expand_objects[expandId]["operations"]]
            #print(operation_frames, len(operation_frames))
            # Find closest frames before and after current frame
            left_frame = None
            for frame in operation_frames:
                if frame <= fr:
                    if left_frame is None or frame > left_frame:
                        left_frame = frame

            # Get expand value
            left_operation = None
            if left_frame is not None:
                left_operation = next(op for op in expand_objects[expandId]["operations"] if op["frame"] == left_frame)
                print("left_operation", left_operation)
                print("fr", fr)
                #print(left_frame, left_frame + int(left_operation["animation_length"]*fps))
                frClamped = max(left_frame, min(fr, left_frame + int(left_operation["animation_length"])))
                #find current frame of the expand movement
                frameValue = c4d.utils.RangeMap(frClamped,
                    left_frame, left_frame + int(left_operation["animation_length"]),
                    left_operation["start_value"], left_operation["dest_value"], False, expandCurve)
                #frameValue = left_operation["dest_value"]
                print("frameValue", frameValue)
            else:
                frameValue = expand_objects[expandId]["initial_value"]
            #get UV coordinates of the hexagon
            hexagon_pts, hex_uvs = create_hex_pts()
            my_uvs = []
            uvHScale = -2.18159
            for i in range(len(hex_uvs)):
                my_uvs.append(c4d.Vector(hex_uvs[i][0]*1., hex_uvs[i][1]*uvHScale, 0.))
            mult = 1 / 50.0
            offUv = c4d.Vector(
                (hexIdx + (ringIdx % 2) * 2.5),
                (ringIdx)*0.757217*uvHScale + 50,
                0.0
            )
            if((hexIdx>47) and (ringIdx % 2 == 1)):
                offUv[0] = offUv[0] - 50.;

            offUv1 = c4d.Vector(0.);
            if((hexIdx==47) and (ringIdx % 2 == 1)):
                offUv1[0] = -50;
            polyUvs = [
                (my_uvs[0] + offUv + offUv1) * mult,
                (my_uvs[3] + offUv + offUv1) * mult,
                (my_uvs[2] + offUv + offUv1) * mult,
                (my_uvs[1] + offUv + offUv1) * mult
            ]
            # Transform UV coordinates (0-1) to pixel coordinates (0-res)
            pixelUvs = []
            for uv in polyUvs:
                pixelUvs.append(c4d.Vector(round(uv.x * res[0]), round(uv.y * res[1]), 0))
            DrawPoly(texture, pixelUvs, c4d.Vector(frameValue*255, 0, 0))

            polyUvs = [
                (my_uvs[0] + offUv) * mult,
                (my_uvs[5] + offUv) * mult,
                (my_uvs[4] + offUv) * mult,
                (my_uvs[3] + offUv) * mult,
            ]
            pixelUvs = []
            for uv in polyUvs:
                pixelUvs.append(c4d.Vector(round(uv.x * res[0]), round(uv.y * res[1]), 0))
            DrawPoly(texture, pixelUvs, c4d.Vector(frameValue*255, 0, 0))

            #print("setsquare", int((res[0]/hexagonsPerRing)*hexIdx), int((res[1]/ringsCount)*ringIdx), int(res[0]//hexagonsPerRing), int(res[1]//ringsCount), c4d.Vector(frameValue*255, 0, 0))
            #DrawSquare(texture,
            #    int((res[0]/hexagonsPerRing)*hexIdx), int(res[1] - int(res[1]/ringsCount)*ringIdx),
            #    int(res[0]/hexagonsPerRing), int(res[1]/ringsCount),
            #    c4d.Vector(hexIdx/hexagonsPerRing*255*0, ringIdx/ringsCount*255*0, frameValue*255))
        texture.Save(filePath, c4d.FILTER_PNG)

def create_hexagon(center_x, center_y, radius):
    points = []
    for i in range(hexPoints):
        angle = math.pi / 3 * i
        x = center_x + radius * math.sin(angle)
        y = center_y + radius * math.cos(angle)
        y *= -1
        if abs(x) > 0.001:
            x = x * 1.149
            y = y * 1.095
        points.append(c4d.Vector(x, y, 0))
    return points

def create_hex_pts():
    w = 0.1487
    h = 0.148593
    hs = 0.0814838
    points = []
    uv_pts = []
    points.append(c4d.Vector(0., -h, 0.))
    points.append(c4d.Vector(w, -hs, 0.))
    points.append(c4d.Vector(w, hs, 0.))
    points.append(c4d.Vector(0, h, 0.))
    points.append(c4d.Vector(-w, hs, 0.))
    points.append(c4d.Vector(-w, -hs, 0.))
    for i in range(len(points)):
        uv = c4d.Vector(points[i])
        uv[0] = uv[0]/w/2. + 0.5
        uv[1] = uv[1]/h/2. + 0.5
        uv[1] *= 48.0/50.0;
        #uv[1] = 1 - uv[1]
        uv_pts.append(uv)
        points[i] *= 100

    return points, uv_pts

def main():

    rootNull = c4d.BaseObject(c4d.Onull)

    rowNulls = []
    scrNulls = []

    visNull = c4d.BaseObject(c4d.Onull)
    visNull.SetName("vis")
    visNull.InsertUnder(rootNull)
    visBox = c4d.BaseObject(c4d.Oplatonic)
    visBox[c4d.PRIM_PLATONIC_TYPE] = c4d.PRIM_PLATONIC_TYPE_OCTA
    visBox[c4d.PRIM_PLATONIC_RAD] = 10
    visBox[c4d.ID_BASEOBJECT_USECOLOR] = c4d.ID_BASEOBJECT_USECOLOR_ALWAYS
    visBox.SetName("visBox")
    visBox.SetRelPos(c4d.Vector(0, 0, 0))
    visBox.SetRelScale(c4d.Vector(0.1))
    #visBox.InsertUnder(visNull)

    contentNull = c4d.BaseObject(c4d.Onull)
    contentNull.SetName("content")
    contentNull.InsertUnder(rootNull)

    parentNull = contentNull
    for i in range(ringsCount):
        rowNull = c4d.BaseObject(c4d.Onull)
        rowNull.SetName(f"row_{i+1}")
        rowNull.SetRelPos(c4d.Vector(0., ringHeight, 0))
        rowNull.InsertUnder(parentNull)
        rowNulls.append(rowNull)

        scrNull = c4d.BaseObject(c4d.Onull)
        scrNull.SetName(f"screens")
        scrNull.InsertUnder(rowNull)
        scrNull.SetRelPos(c4d.Vector(0))
        scrNulls.append(scrNull)

        parentNull = rowNull


    samplePosList = create_sample_positions(ringsCount, hexagonsPerRing, ringHeight, base_circRad)
    field_inputs = prepare_field_inputs(samplePosList, ringsCount, hexCount, ringHeight)
    inputFieldExpand = field_inputs["expand"]
    inputFieldUp = field_inputs["up"]
    inputFieldRot = field_inputs["rotation"]

    sampExpand = fieldListExpand.SampleListSimple(op, inputFieldExpand, c4d.FIELDSAMPLE_FLAG_VALUE)
    sampRot = fieldListRot.SampleListSimple(op, inputFieldRot, c4d.FIELDSAMPLE_FLAG_VALUE)
    sampUp = fieldListUp.SampleListSimple(op, inputFieldUp, c4d.FIELDSAMPLE_FLAG_VALUE)

    sampUpSnapped = [min([0.0, 0.333, 0.666, 1.0], key=lambda x: abs(x - val)) for val in sampUp._value]

    hexagon_template = c4d.PolygonObject(hexPoints, 2)
    hexagon_pts, hex_uvs = create_hex_pts()
    for j, point in enumerate(hexagon_pts):
        hexagon_template.SetPoint(j, point)

    hexagon_template.SetPolygon(0, c4d.CPolygon(0, 3, 2, 1))
    hexagon_template.SetPolygon(1, c4d.CPolygon(0, 5, 4, 3))
    #hexagon_template.MakeVariableTag(c4d.Tuvw,2 )
    #UVWTag = hexagon_template.GetTag(c4d.Tuvw)
    #UVWTag.SetSlow(0, hex_uvs[0], hex_uvs[1], hex_uvs[2], hex_uvs[3])
    #UVWTag.SetSlow(1, hex_uvs[0], hex_uvs[3], hex_uvs[4], hex_uvs[5])


    hexagon_template[c4d.ID_BASEOBJECT_USECOLOR] = c4d.ID_BASEOBJECT_USECOLOR_AUTOMATIC
    #hexagon_template[c4d.ID_BASEOBJECT_COLOR] = c4d.Vector(color_ok)
    my_uvs = []
    uvHScale = -2.18159
    for i in range(len(hex_uvs)):
        my_uvs.append(c4d.Vector(hex_uvs[i][0]*1., hex_uvs[i][1]*uvHScale, 0.))

    for i in range(hexCount):

        ringIdx = i // hexagonsPerRing
        hexIdx = i % hexagonsPerRing
        angle = (2 * math.pi / hexCount) * hexIdx * ringsCount
        angle += (ringIdx % 2) * 0.0628 * 5
        angle += 48.8 * math.pi / 180

        if(displayScreens):
            poly_obj = hexagon_template.GetClone()
            if(colorScreens):
                colorNoise = c4d.Vector(0)
                colMult = 0.2
                colorNoise.x = c4d.utils.noise.Noise(c4d.Vector(i//5)*colMult+0.2)
                colorNoise.y = c4d.utils.noise.Noise(c4d.Vector(i//5)*colMult)
                colorNoise.z = c4d.utils.noise.Noise(c4d.Vector(i//5)*colMult-0.2)
                colorNoise = c4d.utils.RGBToHSV(colorNoise)
                colorNoise.y *= 1.7
                colorNoise = c4d.utils.HSVToRGB(colorNoise)
                poly_obj[c4d.ID_BASEOBJECT_COLOR] = colorNoise

            if(uvScreens):
                poly_obj.MakeVariableTag(c4d.Tuvw, 2)
                UVWTag = poly_obj.GetTag(c4d.Tuvw)
                UVWTag.SetName("uv")
                mult = 1 / 50.0
                offUv = c4d.Vector(
                    (hexIdx + (ringIdx % 2) * 2.5),
                    (ringIdx)*0.757217*uvHScale + 50,
                    0.0
                )
                if((hexIdx>47) and (ringIdx % 2 == 1)):
                    offUv[0] = offUv[0] - 50.;

                offUv1 = c4d.Vector(0.);
                if((hexIdx==47) and (ringIdx % 2 == 1)):
                    offUv1[0] = -50;
                #for i in range(len(hex_uvs)):
                #    my_uvs[i] = c4d.Vector(hex_uvs[i])
                # Set UV coordinates for the first polygon
                UVWTag.SetSlow(
                    0,
                    (my_uvs[0] + offUv + offUv1) * mult,
                    (my_uvs[3] + offUv + offUv1) * mult,
                    (my_uvs[2] + offUv + offUv1) * mult,
                    (my_uvs[1] + offUv + offUv1) * mult
                )

                # Set UV coordinates for the second polygon
                UVWTag.SetSlow(
                    1,
                    (my_uvs[0] + offUv) * mult,
                    (my_uvs[5] + offUv) * mult,
                    (my_uvs[4] + offUv) * mult,
                    (my_uvs[3] + offUv) * mult,
                )

            current_circRad = c4d.utils.RangeMap(sampExpand._value[i//5], 0., 1., diameter_in, diameter_out, True) / 2.

            height = c4d.utils.RangeMap(sampUpSnapped[ringIdx], 0., 1., 0, ringExpansion, False)
            if(ringIdx == 0):
                height = 0.0
            rowNulls[ringIdx].SetRelPos(c4d.Vector(0, height+ringHeight, 0))
            #height = ringIdx * (ringHeight + height)

            base_transform_matrix = c4d.Matrix()
            base_transform_matrix.off = c4d.Vector(
                current_circRad * math.cos(angle),
                0,
                current_circRad * math.sin(angle)
            )

            base_transform_matrix = base_transform_matrix * c4d.utils.MatrixRotY(angle + math.pi / 2.)

            myRot = sampRot._value[i]
            if rotExpand == 0: #clap to 30 degree if Tilt Expand disabled
                myRot = min(max(myRot, 0.17), 0.82)
            rotAngle = c4d.utils.RangeMap(myRot, 0., 1., -45., 45., False) * math.pi / 180
            transform_matrix = base_transform_matrix * c4d.utils.MatrixRotX(rotAngle)

            poly_obj.SetMg(transform_matrix)
            poly_obj.InsertUnderLast(scrNulls[ringIdx])

        if(displayHelpers):
            if((i-2)%5 == 0):
                vis_mg = c4d.Matrix()
                vis_mg.off = c4d.Vector(
                    diameter_in/2. * math.cos(angle),
                    ringHeight*(ringIdx),
                    diameter_in/2. * math.sin(angle)
                )
                visBox = visBox.GetClone()
                visBox.SetMg(vis_mg)
                visBox.InsertUnder(visNull)
            if(colorHelpers):
                visColor = c4d.Vector(sampExpand._value[(i-2)//5], 0, sampUpSnapped[ringIdx])
                visBox[c4d.ID_BASEOBJECT_COLOR] = visColor

    #rootNull.SetDirty(c4d.DIRTYFLAGS_ALL)
    rootNull.Message(c4d.MSG_UPDATE)

    return rootNull