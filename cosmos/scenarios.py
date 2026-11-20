"""Scenario variations for Cosmos 3 transfer, written in NVIDIA's structured prompt schema.

The robot and the manipulated objects come from the edge control video of the
real GR00T episode; the prompt only restyles the world around them. Field names
follow the Cosmos3 transfer cookbook's prompt.json (subjects, background_setting,
lighting, aesthetics, cinematography, ...).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class World:
    key: str
    title: str
    background: str
    lighting: dict[str, str]
    mood: str
    color_scheme: str


WORLDS: dict[str, World] = {world.key: world for world in [
    World(
        "modern_kitchen", "Modern kitchen, daylight",
        "A contemporary home kitchen with white marble countertops, matte black cabinets, brushed steel appliances and a "
        "window letting in morning daylight; a few ceramic dishes and a cutting board sit at the edge of the counter.",
        {"conditions": "Natural morning daylight", "direction": "Soft side light from a window on the left",
         "shadows": "Soft, natural contact shadows under the objects and the arm",
         "illumination_effect": "Clean, bright, true-to-life colors"},
        "Calm, domestic, everyday", "White marble, black cabinetry, brushed steel, warm wood accents"),
    World(
        "warehouse", "Industrial warehouse, fluorescent",
        "An industrial warehouse workstation: a scuffed steel workbench, tall metal shelving with cardboard boxes in "
        "the background, concrete floor and yellow safety markings.",
        {"conditions": "Harsh overhead fluorescent tubes", "direction": "Top-down",
         "shadows": "Hard, short shadows directly beneath objects",
         "illumination_effect": "Cool, slightly greenish industrial light with specular highlights on metal"},
        "Utilitarian, functional, busy", "Grey steel, cardboard brown, safety yellow"),
    World(
        "office", "Office break room, cluttered",
        "A modern office break room counter with a coffee machine, stacked mugs, a laptop, sticky notes and a potted "
        "plant around the workspace; glass meeting-room walls in the background.",
        {"conditions": "Mixed overhead LED panels and screen glow", "direction": "Diffuse from above",
         "shadows": "Soft, low-contrast shadows",
         "illumination_effect": "Even, neutral-white office lighting"},
        "Casual, workplace, slightly cluttered", "Neutral greys, white laminate, green plant, colorful mugs"),
    World(
        "night_kitchen", "Kitchen at night, warm lamp",
        "A cozy rustic kitchen at night with a dark wooden countertop, tiled backsplash, copper pans hanging on the "
        "wall and a dark window reflecting the room.",
        {"conditions": "Warm tungsten pendant lamp at night", "direction": "Single warm source from above right",
         "shadows": "Long, deep, warm-toned shadows",
         "illumination_effect": "Low-key, amber light with a dark surrounding room"},
        "Quiet, intimate, evening", "Amber, dark walnut wood, copper, deep shadow"),
]}


def build_prompt(instruction: str, world: World, num_frames: int, fps: float) -> dict:
    """Cosmos3 structured prompt for one GR00T episode restyled into ``world``."""
    seconds = num_frames / fps
    stamp = f"0:00-0:{int(round(seconds)):02d}"
    robot = (f"A white Franka Emika Panda robot arm with a two-finger gripper performs the task: {instruction}. "
             "It moves smoothly and deliberately, exactly following the motion in the control video.")
    return {
        "subjects": [{
            "description": "A white and dark-grey Franka Emika Panda 7-joint robot arm with a parallel two-finger gripper.",
            "appearance_details": "Glossy white links, dark grey joints and gripper, clean industrial finish.",
            "relationship": "The only agent in the scene, manipulating household objects on the work surface.",
            "location": "Mounted at the back left of the work surface, reaching forward over the objects.",
            "relative_size": "Large within frame",
            "orientation": "Facing the objects on the counter",
            "pose": "Arm extended down toward the work surface",
            "action": instruction,
            "state_changes": f"The arm completes the task '{instruction}' over the course of the clip.",
            "number_of_subjects": 1,
        }],
        "background_setting": world.background,
        "lighting": world.lighting,
        "aesthetics": {"composition": "Fixed three-quarter view of the robot and the work surface",
                       "color_scheme": world.color_scheme, "mood_atmosphere": world.mood, "patterns": ""},
        "cinematography": {"camera_motion": "Static", "framing": "Medium-wide shot", "camera_angle": "Slightly elevated",
                           "depth_of_field": "Deep", "focus": "Sharp focus on the robot arm and the objects",
                           "lens_focal_length": "Standard lens, approximately 35mm equivalent"},
        "style_medium": "Live-action video",
        "artistic_style": "Photorealistic robotics lab footage",
        "context": "A robot arm carrying out a household manipulation task chosen by a brain-computer interface",
        "actions": [{"time": stamp, "description": robot}],
        "text_and_signage_elements": [],
        "segments": [{"segment_index": 0, "time_range": stamp, "description": robot, "key_changes": instruction,
                      "camera": "Static view"}],
        "transitions": [],
        "temporal_caption": robot,
        "audio_description": "",
        "resolution": "640x640",
        "aspect_ratio": "1,1",
        "duration": f"{seconds:.1f}s",
        "fps": fps,
    }
