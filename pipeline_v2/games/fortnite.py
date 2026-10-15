"""Fortnite's word bank: the game's own terms, each with the cue that earns it.

Descriptions were written like scenery captions -- 334 of 935 used words such
as "terrain", "grassy" or "autumn" -- while the things a player actually talks
about (a crack, a box, piece control, a third party) were rarely named. The
taste model learns from these words, so it can only tell clips apart by what
they mention.

Every term comes with the visible cue that justifies it, because a bare list
invites a model to sprinkle jargon to sound expert ("edit reset" with no edit
in sight). Terms are season-stable on purpose: mechanics and HUD events, not
item names, which the model reads off the screen instead.

A second game gets its own module with the same two names.
"""

from __future__ import annotations

# Allowed `mechanics` tags, beyond the default list. Kept short and countable:
# each is something the ranker can use as a feature.
MECHANICS = (
    "editing",
    "piece_control",
    "box_fight",
    "shield_crack",
    "third_party",
    "multi_elimination",
)

WORD_BANK = """
Use Fortnite's own terms. Use a term ONLY when the frames show the cue given
for it; otherwise describe plainly. Prefer the term to a paraphrase.

Combat
- elimination: an ELIMINATED banner or the player's kill-feed line names the
  opponent. Name the opponent if the banner or feed shows it.
- knock: the opponent drops to a crawling, downed state instead of dying
  (duos/squads). In solos a lethal hit is an elimination, not a knock.
- finish: eliminating an opponent who was already knocked.
- shield crack (crack): the opponent's shield breaks -- a blue shatter effect,
  blue damage numbers turning white. Give the damage number if legible.
- one-pump: one shotgun shot takes the opponent from full to eliminated/knocked.
- headshot: a headshot damage number (larger, yellow/orange) on the hit.
- tag: a small long-range hit that does not finish anything.
- spray: sustained automatic fire from an SMG or assault rifle.
- multi-elimination: two or more eliminations by the player in these frames.
- third party: a player or team not in the original fight attacks during it.
- damage numbers: report the values shown; they are the clearest evidence.

Building and editing
- build fight: both sides placing builds at each other.
- box / box up: four walls (often a roof) placed around the player.
- ramp rush: pushing toward an opponent by placing ramps forward and up.
- high ground / high-ground retake: being above the opponent / taking it back.
- piece control: placing pieces that trap or cut off an opponent's builds.
- edit: the edit grid appears, or a wall, floor or ramp changes shape
  (window, door, half-wall, peek). edit reset: the edited piece restored.
- take a wall / take a piece: building into a spot the opponent was building.
- 90s: the turning ramp-and-wall climb, repeated.
- tunnel, cone, turtle: say so only if that exact structure is built.

Movement and match flow
- pre-game island (warm-up): before the Battle Bus; kills there do not count,
  so the outcome is no_event.
- drop / landing: skydiving or gliding from the Battle Bus to the ground.
- rotate (rotation): travelling toward the safe zone.
- storm / zone: the purple storm wall or its timer; moving zone, final circle.
- loot: picking up items, chests, ammo boxes, floor loot. heal: using medkits,
  shield potions, Chug Splashes or similar.
- reboot: bringing a teammate back at a Reboot Van.
- spectating: the camera follows another player after the player's death.
- Victory Royale: the VICTORY ROYALE banner.

Weapons
- Read the weapon's name from the HUD or kill feed. Rarity by colour: grey
  common, green uncommon, blue rare, purple epic, gold legendary, mythic.
- If the name is unreadable, use the family: shotgun, SMG, assault rifle,
  pistol, sniper rifle, DMR. Never guess a specific name.

Never describe scenery, weather, foliage, colours, lighting, skins or
cosmetics. Mention the surroundings only when they change the fight: cover,
high ground, the storm, a named location shown on the HUD.
"""
