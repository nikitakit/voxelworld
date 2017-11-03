const inertBlockProps = {
    missing: {texture: 'no_texture', displayName: 'Missing Block'}, // custom texture (TODO: standard location?)

    // Hand-coded values for some block types
    waterFlow: {texture: 'water_flow'}, // TODO: animation
    water: {texture: 'water_still'}, // TODO: animation
    lavaFlow: {texture: 'lava_flow'}, // TODO: animation
    lava: {texture: 'lava_still'} // TODO: animation
}

const mcBlockID2Voxel = {
  default: 'missing',

  // hand-coded values for some block types
  '8:0': 'waterFlow',
  '9:0': 'water',
  '10:0': 'lavaFlow',
  '11:0': 'lava'
};

const rawData = [
// BEGIN JSON OBJECT
{
 "35:4": {
  "right": "wool_colored_yellow",
  "back": "wool_colored_yellow",
  "top": "wool_colored_yellow",
  "front": "wool_colored_yellow",
  "bottom": "wool_colored_yellow",
  "left": "wool_colored_yellow"
 },
 "97:5": {
  "right": "stonebrick_carved",
  "back": "stonebrick_carved",
  "top": "stonebrick_carved",
  "front": "stonebrick_carved",
  "bottom": "stonebrick_carved",
  "left": "stonebrick_carved"
 },
 "212:3": {
  "right": "frosted_ice_3",
  "back": "frosted_ice_3",
  "top": "frosted_ice_3",
  "front": "frosted_ice_3",
  "bottom": "frosted_ice_3",
  "left": "frosted_ice_3"
 },
 "43:8": {
  "right": "stone_slab_top",
  "back": "stone_slab_top",
  "top": "stone_slab_top",
  "front": "stone_slab_top",
  "bottom": "stone_slab_top",
  "left": "stone_slab_top"
 },
 "158:0": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dropper_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "35:11": {
  "right": "wool_colored_blue",
  "back": "wool_colored_blue",
  "top": "wool_colored_blue",
  "front": "wool_colored_blue",
  "bottom": "wool_colored_blue",
  "left": "wool_colored_blue"
 },
 "18:14": {
  "right": "leaves_birch",
  "back": "leaves_birch",
  "top": "leaves_birch",
  "front": "leaves_birch",
  "bottom": "leaves_birch",
  "left": "leaves_birch"
 },
 "159:14": {
  "right": "hardened_clay_stained_red",
  "back": "hardened_clay_stained_red",
  "top": "hardened_clay_stained_red",
  "front": "hardened_clay_stained_red",
  "bottom": "hardened_clay_stained_red",
  "left": "hardened_clay_stained_red"
 },
 "84:1": {
  "right": "jukebox_side",
  "back": "jukebox_side",
  "top": "jukebox_top",
  "front": "jukebox_side",
  "bottom": "jukebox_side",
  "left": "jukebox_side"
 },
 "17:8": {
  "right": "log_oak",
  "back": "log_oak",
  "top": "log_oak_top",
  "front": "log_oak",
  "bottom": "log_oak_top",
  "left": "log_oak"
 },
 "23:10": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "100:15": {
  "right": "mushroom_block_skin_stem",
  "back": "mushroom_block_skin_stem",
  "top": "mushroom_block_skin_stem",
  "front": "mushroom_block_skin_stem",
  "bottom": "mushroom_block_skin_stem",
  "left": "mushroom_block_skin_stem"
 },
 "35:0": {
  "right": "wool_colored_white",
  "back": "wool_colored_white",
  "top": "wool_colored_white",
  "front": "wool_colored_white",
  "bottom": "wool_colored_white",
  "left": "wool_colored_white"
 },
 "18:7": {
  "right": "leaves_jungle",
  "back": "leaves_jungle",
  "top": "leaves_jungle",
  "front": "leaves_jungle",
  "bottom": "leaves_jungle",
  "left": "leaves_jungle"
 },
 "204:0": {
  "right": "purpur_block",
  "back": "purpur_block",
  "top": "purpur_block",
  "front": "purpur_block",
  "bottom": "purpur_block",
  "left": "purpur_block"
 },
 "17:6": {
  "right": "log_birch",
  "back": "log_birch",
  "top": "log_birch_top",
  "front": "log_birch",
  "bottom": "log_birch_top",
  "left": "log_birch"
 },
 "29:3": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_sticky"
 },
 "35:3": {
  "right": "wool_colored_light_blue",
  "back": "wool_colored_light_blue",
  "top": "wool_colored_light_blue",
  "front": "wool_colored_light_blue",
  "bottom": "wool_colored_light_blue",
  "left": "wool_colored_light_blue"
 },
 "158:12": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "100:9": {
  "right": "mushroom_block_skin_red",
  "back": "mushroom_block_skin_red",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "91:2": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_on"
 },
 "162:13": {
  "right": "log_big_oak",
  "back": "log_big_oak",
  "top": "log_big_oak",
  "front": "log_big_oak",
  "bottom": "log_big_oak",
  "left": "log_big_oak"
 },
 "153:0": {
  "right": "quartz_ore",
  "back": "quartz_ore",
  "top": "quartz_ore",
  "front": "quartz_ore",
  "bottom": "quartz_ore",
  "left": "quartz_ore"
 },
 "1:3": {
  "right": "stone_diorite",
  "back": "stone_diorite",
  "top": "stone_diorite",
  "front": "stone_diorite",
  "bottom": "stone_diorite",
  "left": "stone_diorite"
 },
 "159:15": {
  "right": "hardened_clay_stained_black",
  "back": "hardened_clay_stained_black",
  "top": "hardened_clay_stained_black",
  "front": "hardened_clay_stained_black",
  "bottom": "hardened_clay_stained_black",
  "left": "hardened_clay_stained_black"
 },
 "5:5": {
  "right": "planks_big_oak",
  "back": "planks_big_oak",
  "top": "planks_big_oak",
  "front": "planks_big_oak",
  "bottom": "planks_big_oak",
  "left": "planks_big_oak"
 },
 "170:0": {
  "right": "hay_block_side",
  "back": "hay_block_side",
  "top": "hay_block_top",
  "front": "hay_block_side",
  "bottom": "hay_block_top",
  "left": "hay_block_side"
 },
 "23:13": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "43:7": {
  "right": "quartz_block_side",
  "back": "quartz_block_side",
  "top": "quartz_block_top",
  "front": "quartz_block_side",
  "bottom": "quartz_block_bottom",
  "left": "quartz_block_side"
 },
 "91:0": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_on"
 },
 "212:0": {
  "right": "frosted_ice_0",
  "back": "frosted_ice_0",
  "top": "frosted_ice_0",
  "front": "frosted_ice_0",
  "bottom": "frosted_ice_0",
  "left": "frosted_ice_0"
 },
 "17:7": {
  "right": "log_jungle",
  "back": "log_jungle",
  "top": "log_jungle_top",
  "front": "log_jungle",
  "bottom": "log_jungle_top",
  "left": "log_jungle"
 },
 "95:13": {
  "right": "glass_green",
  "back": "glass_green",
  "top": "glass_green",
  "front": "glass_green",
  "bottom": "glass_green",
  "left": "glass_green"
 },
 "158:10": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "210:8": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_conditional",
  "top": "repeating_command_block_conditional",
  "front": "repeating_command_block_conditional",
  "bottom": "repeating_command_block_conditional",
  "left": "repeating_command_block_front"
 },
 "17:11": {
  "right": "log_jungle",
  "back": "log_jungle",
  "top": "log_jungle_top",
  "front": "log_jungle",
  "bottom": "log_jungle_top",
  "left": "log_jungle"
 },
 "174:0": {
  "right": "ice_packed",
  "back": "ice_packed",
  "top": "ice_packed",
  "front": "ice_packed",
  "bottom": "ice_packed",
  "left": "ice_packed"
 },
 "206:0": {
  "right": "end_bricks",
  "back": "end_bricks",
  "top": "end_bricks",
  "front": "end_bricks",
  "bottom": "end_bricks",
  "left": "end_bricks"
 },
 "18:8": {
  "right": "leaves_oak",
  "back": "leaves_oak",
  "top": "leaves_oak",
  "front": "leaves_oak",
  "bottom": "leaves_oak",
  "left": "leaves_oak"
 },
 "169:0": {
  "right": "sea_lantern",
  "back": "sea_lantern",
  "top": "sea_lantern",
  "front": "sea_lantern",
  "bottom": "sea_lantern",
  "left": "sea_lantern"
 },
 "35:10": {
  "right": "wool_colored_purple",
  "back": "wool_colored_purple",
  "top": "wool_colored_purple",
  "front": "wool_colored_purple",
  "bottom": "wool_colored_purple",
  "left": "wool_colored_purple"
 },
 "5:0": {
  "right": "planks_oak",
  "back": "planks_oak",
  "top": "planks_oak",
  "front": "planks_oak",
  "bottom": "planks_oak",
  "left": "planks_oak"
 },
 "62:0": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_on"
 },
 "99:14": {
  "right": "mushroom_block_skin_brown",
  "back": "mushroom_block_skin_brown",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_skin_brown",
  "bottom": "mushroom_block_skin_brown",
  "left": "mushroom_block_skin_brown"
 },
 "212:2": {
  "right": "frosted_ice_2",
  "back": "frosted_ice_2",
  "top": "frosted_ice_2",
  "front": "frosted_ice_2",
  "bottom": "frosted_ice_2",
  "left": "frosted_ice_2"
 },
 "99:3": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_skin_brown",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_brown"
 },
 "100:0": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_inside",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "100:7": {
  "right": "mushroom_block_skin_red",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_skin_red",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "212:1": {
  "right": "frosted_ice_1",
  "back": "frosted_ice_1",
  "top": "frosted_ice_1",
  "front": "frosted_ice_1",
  "bottom": "frosted_ice_1",
  "left": "frosted_ice_1"
 },
 "110:0": {
  "right": "mycelium_side",
  "back": "mycelium_side",
  "top": "mycelium_top",
  "front": "mycelium_side",
  "bottom": "dirt",
  "left": "mycelium_side"
 },
 "179:0": {
  "right": "red_sandstone_normal",
  "back": "red_sandstone_normal",
  "top": "red_sandstone_top",
  "front": "red_sandstone_normal",
  "bottom": "red_sandstone_bottom",
  "left": "red_sandstone_normal"
 },
 "18:11": {
  "right": "leaves_jungle",
  "back": "leaves_jungle",
  "top": "leaves_jungle",
  "front": "leaves_jungle",
  "bottom": "leaves_jungle",
  "left": "leaves_jungle"
 },
 "46:1": {
  "right": "tnt_side",
  "back": "tnt_side",
  "top": "tnt_top",
  "front": "tnt_side",
  "bottom": "tnt_bottom",
  "left": "tnt_side"
 },
 "4:0": {
  "right": "cobblestone",
  "back": "cobblestone",
  "top": "cobblestone",
  "front": "cobblestone",
  "bottom": "cobblestone",
  "left": "cobblestone"
 },
 "99:7": {
  "right": "mushroom_block_skin_brown",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_skin_brown",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "179:2": {
  "right": "red_sandstone_smooth",
  "back": "red_sandstone_smooth",
  "top": "red_sandstone_top",
  "front": "red_sandstone_smooth",
  "bottom": "red_sandstone_top",
  "left": "red_sandstone_smooth"
 },
 "17:13": {
  "right": "log_spruce",
  "back": "log_spruce",
  "top": "log_spruce",
  "front": "log_spruce",
  "bottom": "log_spruce",
  "left": "log_spruce"
 },
 "168:0": {
  "right": "prismarine_rough",
  "back": "prismarine_rough",
  "top": "prismarine_rough",
  "front": "prismarine_rough",
  "bottom": "prismarine_rough",
  "left": "prismarine_rough"
 },
 "56:0": {
  "right": "diamond_ore",
  "back": "diamond_ore",
  "top": "diamond_ore",
  "front": "diamond_ore",
  "bottom": "diamond_ore",
  "left": "diamond_ore"
 },
 "79:0": {
  "right": "ice",
  "back": "ice",
  "top": "ice",
  "front": "ice",
  "bottom": "ice",
  "left": "ice"
 },
 "137:0": {
  "right": "command_block_back",
  "back": "command_block_side",
  "top": "command_block_side",
  "front": "command_block_side",
  "bottom": "command_block_side",
  "left": "command_block_front"
 },
 "162:12": {
  "right": "log_acacia",
  "back": "log_acacia",
  "top": "log_acacia",
  "front": "log_acacia",
  "bottom": "log_acacia",
  "left": "log_acacia"
 },
 "159:3": {
  "right": "hardened_clay_stained_light_blue",
  "back": "hardened_clay_stained_light_blue",
  "top": "hardened_clay_stained_light_blue",
  "front": "hardened_clay_stained_light_blue",
  "bottom": "hardened_clay_stained_light_blue",
  "left": "hardened_clay_stained_light_blue"
 },
 "19:0": {
  "right": "sponge",
  "back": "sponge",
  "top": "sponge",
  "front": "sponge",
  "bottom": "sponge",
  "left": "sponge"
 },
 "84:0": {
  "right": "jukebox_side",
  "back": "jukebox_side",
  "top": "jukebox_top",
  "front": "jukebox_side",
  "bottom": "jukebox_side",
  "left": "jukebox_side"
 },
 "95:0": {
  "right": "glass_white",
  "back": "glass_white",
  "top": "glass_white",
  "front": "glass_white",
  "bottom": "glass_white",
  "left": "glass_white"
 },
 "95:1": {
  "right": "glass_orange",
  "back": "glass_orange",
  "top": "glass_orange",
  "front": "glass_orange",
  "bottom": "glass_orange",
  "left": "glass_orange"
 },
 "159:12": {
  "right": "hardened_clay_stained_brown",
  "back": "hardened_clay_stained_brown",
  "top": "hardened_clay_stained_brown",
  "front": "hardened_clay_stained_brown",
  "bottom": "hardened_clay_stained_brown",
  "left": "hardened_clay_stained_brown"
 },
 "155:1": {
  "right": "quartz_block_chiseled",
  "back": "quartz_block_chiseled",
  "top": "quartz_block_chiseled_top",
  "front": "quartz_block_chiseled",
  "bottom": "quartz_block_chiseled_top",
  "left": "quartz_block_chiseled"
 },
 "137:3": {
  "right": "command_block_back",
  "back": "command_block_side",
  "top": "command_block_side",
  "front": "command_block_side",
  "bottom": "command_block_side",
  "left": "command_block_front"
 },
 "158:9": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dropper_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "161:12": {
  "right": "leaves_acacia",
  "back": "leaves_acacia",
  "top": "leaves_acacia",
  "front": "leaves_acacia",
  "bottom": "leaves_acacia",
  "left": "leaves_acacia"
 },
 "18:10": {
  "right": "leaves_birch",
  "back": "leaves_birch",
  "top": "leaves_birch",
  "front": "leaves_birch",
  "bottom": "leaves_birch",
  "left": "leaves_birch"
 },
 "159:2": {
  "right": "hardened_clay_stained_magenta",
  "back": "hardened_clay_stained_magenta",
  "top": "hardened_clay_stained_magenta",
  "front": "hardened_clay_stained_magenta",
  "bottom": "hardened_clay_stained_magenta",
  "left": "hardened_clay_stained_magenta"
 },
 "17:5": {
  "right": "log_spruce",
  "back": "log_spruce",
  "top": "log_spruce_top",
  "front": "log_spruce",
  "bottom": "log_spruce_top",
  "left": "log_spruce"
 },
 "95:4": {
  "right": "glass_yellow",
  "back": "glass_yellow",
  "top": "glass_yellow",
  "front": "glass_yellow",
  "bottom": "glass_yellow",
  "left": "glass_yellow"
 },
 "129:0": {
  "right": "emerald_ore",
  "back": "emerald_ore",
  "top": "emerald_ore",
  "front": "emerald_ore",
  "bottom": "emerald_ore",
  "left": "emerald_ore"
 },
 "18:5": {
  "right": "leaves_spruce",
  "back": "leaves_spruce",
  "top": "leaves_spruce",
  "front": "leaves_spruce",
  "bottom": "leaves_spruce",
  "left": "leaves_spruce"
 },
 "95:8": {
  "right": "glass_silver",
  "back": "glass_silver",
  "top": "glass_silver",
  "front": "glass_silver",
  "bottom": "glass_silver",
  "left": "glass_silver"
 },
 "23:5": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "18:0": {
  "right": "leaves_oak",
  "back": "leaves_oak",
  "top": "leaves_oak",
  "front": "leaves_oak",
  "bottom": "leaves_oak",
  "left": "leaves_oak"
 },
 "211:12": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_conditional",
  "top": "chain_command_block_conditional",
  "front": "chain_command_block_conditional",
  "bottom": "chain_command_block_conditional",
  "left": "chain_command_block_front"
 },
 "21:0": {
  "right": "lapis_ore",
  "back": "lapis_ore",
  "top": "lapis_ore",
  "front": "lapis_ore",
  "bottom": "lapis_ore",
  "left": "lapis_ore"
 },
 "23:9": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dispenser_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "24:0": {
  "right": "sandstone_normal",
  "back": "sandstone_normal",
  "top": "sandstone_top",
  "front": "sandstone_normal",
  "bottom": "sandstone_bottom",
  "left": "sandstone_normal"
 },
 "43:9": {
  "right": "sandstone_top",
  "back": "sandstone_top",
  "top": "sandstone_top",
  "front": "sandstone_top",
  "bottom": "sandstone_top",
  "left": "sandstone_top"
 },
 "17:3": {
  "right": "log_jungle",
  "back": "log_jungle",
  "top": "log_jungle_top",
  "front": "log_jungle",
  "bottom": "log_jungle_top",
  "left": "log_jungle"
 },
 "98:0": {
  "right": "stonebrick",
  "back": "stonebrick",
  "top": "stonebrick",
  "front": "stonebrick",
  "bottom": "stonebrick",
  "left": "stonebrick"
 },
 "18:9": {
  "right": "leaves_spruce",
  "back": "leaves_spruce",
  "top": "leaves_spruce",
  "front": "leaves_spruce",
  "bottom": "leaves_spruce",
  "left": "leaves_spruce"
 },
 "13:0": {
  "right": "gravel",
  "back": "gravel",
  "top": "gravel",
  "front": "gravel",
  "bottom": "gravel",
  "left": "gravel"
 },
 "35:7": {
  "right": "wool_colored_gray",
  "back": "wool_colored_gray",
  "top": "wool_colored_gray",
  "front": "wool_colored_gray",
  "bottom": "wool_colored_gray",
  "left": "wool_colored_gray"
 },
 "121:0": {
  "right": "end_stone",
  "back": "end_stone",
  "top": "end_stone",
  "front": "end_stone",
  "bottom": "end_stone",
  "left": "end_stone"
 },
 "98:1": {
  "right": "stonebrick_mossy",
  "back": "stonebrick_mossy",
  "top": "stonebrick_mossy",
  "front": "stonebrick_mossy",
  "bottom": "stonebrick_mossy",
  "left": "stonebrick_mossy"
 },
 "124:0": {
  "right": "redstone_lamp_on",
  "back": "redstone_lamp_on",
  "top": "redstone_lamp_on",
  "front": "redstone_lamp_on",
  "bottom": "redstone_lamp_on",
  "left": "redstone_lamp_on"
 },
 "3:1": {
  "right": "coarse_dirt",
  "back": "coarse_dirt",
  "top": "coarse_dirt",
  "front": "coarse_dirt",
  "bottom": "coarse_dirt",
  "left": "coarse_dirt"
 },
 "255:0": {
  "right": "structure_block_save",
  "back": "structure_block_save",
  "top": "structure_block_save",
  "front": "structure_block_save",
  "bottom": "structure_block_save",
  "left": "structure_block_save"
 },
 "45:0": {
  "right": "brick",
  "back": "brick",
  "top": "brick",
  "front": "brick",
  "bottom": "brick",
  "left": "brick"
 },
 "33:3": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_normal"
 },
 "99:4": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_skin_brown",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "89:0": {
  "right": "glowstone",
  "back": "glowstone",
  "top": "glowstone",
  "front": "glowstone",
  "bottom": "glowstone",
  "left": "glowstone"
 },
 "95:11": {
  "right": "glass_blue",
  "back": "glass_blue",
  "top": "glass_blue",
  "front": "glass_blue",
  "bottom": "glass_blue",
  "left": "glass_blue"
 },
 "170:8": {
  "right": "hay_block_side",
  "back": "hay_block_side",
  "top": "hay_block_top",
  "front": "hay_block_side",
  "bottom": "hay_block_top",
  "left": "hay_block_side"
 },
 "24:2": {
  "right": "sandstone_smooth",
  "back": "sandstone_smooth",
  "top": "sandstone_top",
  "front": "sandstone_smooth",
  "bottom": "sandstone_top",
  "left": "sandstone_smooth"
 },
 "18:6": {
  "right": "leaves_birch",
  "back": "leaves_birch",
  "top": "leaves_birch",
  "front": "leaves_birch",
  "bottom": "leaves_birch",
  "left": "leaves_birch"
 },
 "16:0": {
  "right": "coal_ore",
  "back": "coal_ore",
  "top": "coal_ore",
  "front": "coal_ore",
  "bottom": "coal_ore",
  "left": "coal_ore"
 },
 "18:4": {
  "right": "leaves_oak",
  "back": "leaves_oak",
  "top": "leaves_oak",
  "front": "leaves_oak",
  "bottom": "leaves_oak",
  "left": "leaves_oak"
 },
 "100:10": {
  "right": "mushroom_block_skin_stem",
  "back": "mushroom_block_skin_stem",
  "top": "mushroom_block_inside",
  "front": "mushroom_block_skin_stem",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_stem"
 },
 "91:3": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_on"
 },
 "159:5": {
  "right": "hardened_clay_stained_lime",
  "back": "hardened_clay_stained_lime",
  "top": "hardened_clay_stained_lime",
  "front": "hardened_clay_stained_lime",
  "bottom": "hardened_clay_stained_lime",
  "left": "hardened_clay_stained_lime"
 },
 "43:6": {
  "right": "nether_brick",
  "back": "nether_brick",
  "top": "nether_brick",
  "front": "nether_brick",
  "bottom": "nether_brick",
  "left": "nether_brick"
 },
 "173:0": {
  "right": "coal_block",
  "back": "coal_block",
  "top": "coal_block",
  "front": "coal_block",
  "bottom": "coal_block",
  "left": "coal_block"
 },
 "5:2": {
  "right": "planks_birch",
  "back": "planks_birch",
  "top": "planks_birch",
  "front": "planks_birch",
  "bottom": "planks_birch",
  "left": "planks_birch"
 },
 "100:6": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_skin_red",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "17:10": {
  "right": "log_birch",
  "back": "log_birch",
  "top": "log_birch_top",
  "front": "log_birch",
  "bottom": "log_birch_top",
  "left": "log_birch"
 },
 "61:5": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_off"
 },
 "99:9": {
  "right": "mushroom_block_skin_brown",
  "back": "mushroom_block_skin_brown",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "52:0": {
  "right": "mob_spawner",
  "back": "mob_spawner",
  "top": "mob_spawner",
  "front": "mob_spawner",
  "bottom": "mob_spawner",
  "left": "mob_spawner"
 },
 "152:0": {
  "right": "redstone_block",
  "back": "redstone_block",
  "top": "redstone_block",
  "front": "redstone_block",
  "bottom": "redstone_block",
  "left": "redstone_block"
 },
 "162:4": {
  "right": "log_acacia",
  "back": "log_acacia",
  "top": "log_acacia_top",
  "front": "log_acacia",
  "bottom": "log_acacia_top",
  "left": "log_acacia"
 },
 "86:2": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_off"
 },
 "18:15": {
  "right": "leaves_jungle",
  "back": "leaves_jungle",
  "top": "leaves_jungle",
  "front": "leaves_jungle",
  "bottom": "leaves_jungle",
  "left": "leaves_jungle"
 },
 "100:3": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_skin_red",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_red"
 },
 "95:2": {
  "right": "glass_magenta",
  "back": "glass_magenta",
  "top": "glass_magenta",
  "front": "glass_magenta",
  "bottom": "glass_magenta",
  "left": "glass_magenta"
 },
 "57:0": {
  "right": "diamond_block",
  "back": "diamond_block",
  "top": "diamond_block",
  "front": "diamond_block",
  "bottom": "diamond_block",
  "left": "diamond_block"
 },
 "158:11": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "158:13": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "159:9": {
  "right": "hardened_clay_stained_cyan",
  "back": "hardened_clay_stained_cyan",
  "top": "hardened_clay_stained_cyan",
  "front": "hardened_clay_stained_cyan",
  "bottom": "hardened_clay_stained_cyan",
  "left": "hardened_clay_stained_cyan"
 },
 "82:0": {
  "right": "clay",
  "back": "clay",
  "top": "clay",
  "front": "clay",
  "bottom": "clay",
  "left": "clay"
 },
 "137:5": {
  "right": "command_block_back",
  "back": "command_block_side",
  "top": "command_block_side",
  "front": "command_block_side",
  "bottom": "command_block_side",
  "left": "command_block_front"
 },
 "125:4": {
  "right": "planks_acacia",
  "back": "planks_acacia",
  "top": "planks_acacia",
  "front": "planks_acacia",
  "bottom": "planks_acacia",
  "left": "planks_acacia"
 },
 "86:3": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_off"
 },
 "202:8": {
  "right": "purpur_pillar",
  "back": "purpur_pillar",
  "top": "purpur_pillar_top",
  "front": "purpur_pillar",
  "bottom": "purpur_pillar_top",
  "left": "purpur_pillar"
 },
 "161:13": {
  "right": "leaves_big_oak",
  "back": "leaves_big_oak",
  "top": "leaves_big_oak",
  "front": "leaves_big_oak",
  "bottom": "leaves_big_oak",
  "left": "leaves_big_oak"
 },
 "125:2": {
  "right": "planks_birch",
  "back": "planks_birch",
  "top": "planks_birch",
  "front": "planks_birch",
  "bottom": "planks_birch",
  "left": "planks_birch"
 },
 "210:4": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_side",
  "top": "repeating_command_block_side",
  "front": "repeating_command_block_side",
  "bottom": "repeating_command_block_side",
  "left": "repeating_command_block_front"
 },
 "43:2": {
  "right": "planks_oak",
  "back": "planks_oak",
  "top": "planks_oak",
  "front": "planks_oak",
  "bottom": "planks_oak",
  "left": "planks_oak"
 },
 "43:3": {
  "right": "cobblestone",
  "back": "cobblestone",
  "top": "cobblestone",
  "front": "cobblestone",
  "bottom": "cobblestone",
  "left": "cobblestone"
 },
 "14:0": {
  "right": "gold_ore",
  "back": "gold_ore",
  "top": "gold_ore",
  "front": "gold_ore",
  "bottom": "gold_ore",
  "left": "gold_ore"
 },
 "61:4": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_off"
 },
 "12:0": {
  "right": "sand",
  "back": "sand",
  "top": "sand",
  "front": "sand",
  "bottom": "sand",
  "left": "sand"
 },
 "125:3": {
  "right": "planks_jungle",
  "back": "planks_jungle",
  "top": "planks_jungle",
  "front": "planks_jungle",
  "bottom": "planks_jungle",
  "left": "planks_jungle"
 },
 "29:0": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_sticky"
 },
 "19:1": {
  "right": "sponge_wet",
  "back": "sponge_wet",
  "top": "sponge_wet",
  "front": "sponge_wet",
  "bottom": "sponge_wet",
  "left": "sponge_wet"
 },
 "211:3": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_side",
  "top": "chain_command_block_side",
  "front": "chain_command_block_side",
  "bottom": "chain_command_block_side",
  "left": "chain_command_block_front"
 },
 "99:0": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_inside",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "1:0": {
  "right": "stone",
  "back": "stone",
  "top": "stone",
  "front": "stone",
  "bottom": "stone",
  "left": "stone"
 },
 "35:15": {
  "right": "wool_colored_black",
  "back": "wool_colored_black",
  "top": "wool_colored_black",
  "front": "wool_colored_black",
  "bottom": "wool_colored_black",
  "left": "wool_colored_black"
 },
 "29:5": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_sticky"
 },
 "211:10": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_conditional",
  "top": "chain_command_block_conditional",
  "front": "chain_command_block_conditional",
  "bottom": "chain_command_block_conditional",
  "left": "chain_command_block_front"
 },
 "158:2": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "33:2": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_normal"
 },
 "18:1": {
  "right": "leaves_spruce",
  "back": "leaves_spruce",
  "top": "leaves_spruce",
  "front": "leaves_spruce",
  "bottom": "leaves_spruce",
  "left": "leaves_spruce"
 },
 "158:5": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "18:12": {
  "right": "leaves_oak",
  "back": "leaves_oak",
  "top": "leaves_oak",
  "front": "leaves_oak",
  "bottom": "leaves_oak",
  "left": "leaves_oak"
 },
 "80:0": {
  "right": "snow",
  "back": "snow",
  "top": "snow",
  "front": "snow",
  "bottom": "snow",
  "left": "snow"
 },
 "211:8": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_conditional",
  "top": "chain_command_block_conditional",
  "front": "chain_command_block_conditional",
  "bottom": "chain_command_block_conditional",
  "left": "chain_command_block_front"
 },
 "23:1": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dispenser_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "23:11": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "210:9": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_conditional",
  "top": "repeating_command_block_conditional",
  "front": "repeating_command_block_conditional",
  "bottom": "repeating_command_block_conditional",
  "left": "repeating_command_block_front"
 },
 "43:13": {
  "right": "stonebrick",
  "back": "stonebrick",
  "top": "stonebrick",
  "front": "stonebrick",
  "bottom": "stonebrick",
  "left": "stonebrick"
 },
 "97:1": {
  "right": "cobblestone",
  "back": "cobblestone",
  "top": "cobblestone",
  "front": "cobblestone",
  "bottom": "cobblestone",
  "left": "cobblestone"
 },
 "1:4": {
  "right": "stone_diorite_smooth",
  "back": "stone_diorite_smooth",
  "top": "stone_diorite_smooth",
  "front": "stone_diorite_smooth",
  "bottom": "stone_diorite_smooth",
  "left": "stone_diorite_smooth"
 },
 "210:11": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_conditional",
  "top": "repeating_command_block_conditional",
  "front": "repeating_command_block_conditional",
  "bottom": "repeating_command_block_conditional",
  "left": "repeating_command_block_front"
 },
 "210:1": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_side",
  "top": "repeating_command_block_side",
  "front": "repeating_command_block_side",
  "bottom": "repeating_command_block_side",
  "left": "repeating_command_block_front"
 },
 "35:8": {
  "right": "wool_colored_silver",
  "back": "wool_colored_silver",
  "top": "wool_colored_silver",
  "front": "wool_colored_silver",
  "bottom": "wool_colored_silver",
  "left": "wool_colored_silver"
 },
 "23:4": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "155:0": {
  "right": "quartz_block_side",
  "back": "quartz_block_side",
  "top": "quartz_block_top",
  "front": "quartz_block_side",
  "bottom": "quartz_block_bottom",
  "left": "quartz_block_side"
 },
 "3:2": {
  "right": "dirt_podzol_side",
  "back": "dirt_podzol_side",
  "top": "dirt_podzol_top",
  "front": "dirt_podzol_side",
  "bottom": "dirt",
  "left": "dirt_podzol_side"
 },
 "99:1": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_skin_brown",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_brown"
 },
 "137:12": {
  "right": "command_block_back",
  "back": "command_block_conditional",
  "top": "command_block_conditional",
  "front": "command_block_conditional",
  "bottom": "command_block_conditional",
  "left": "command_block_front"
 },
 "112:0": {
  "right": "nether_brick",
  "back": "nether_brick",
  "top": "nether_brick",
  "front": "nether_brick",
  "bottom": "nether_brick",
  "left": "nether_brick"
 },
 "61:3": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_off"
 },
 "43:4": {
  "right": "brick",
  "back": "brick",
  "top": "brick",
  "front": "brick",
  "bottom": "brick",
  "left": "brick"
 },
 "33:0": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_normal"
 },
 "24:1": {
  "right": "sandstone_carved",
  "back": "sandstone_carved",
  "top": "sandstone_top",
  "front": "sandstone_carved",
  "bottom": "sandstone_top",
  "left": "sandstone_carved"
 },
 "125:5": {
  "right": "planks_big_oak",
  "back": "planks_big_oak",
  "top": "planks_big_oak",
  "front": "planks_big_oak",
  "bottom": "planks_big_oak",
  "left": "planks_big_oak"
 },
 "20:0": {
  "right": "glass",
  "back": "glass",
  "top": "glass",
  "front": "glass",
  "bottom": "glass",
  "left": "glass"
 },
 "73:0": {
  "right": "redstone_ore",
  "back": "redstone_ore",
  "top": "redstone_ore",
  "front": "redstone_ore",
  "bottom": "redstone_ore",
  "left": "redstone_ore"
 },
 "133:0": {
  "right": "emerald_block",
  "back": "emerald_block",
  "top": "emerald_block",
  "front": "emerald_block",
  "bottom": "emerald_block",
  "left": "emerald_block"
 },
 "172:0": {
  "right": "hardened_clay",
  "back": "hardened_clay",
  "top": "hardened_clay",
  "front": "hardened_clay",
  "bottom": "hardened_clay",
  "left": "hardened_clay"
 },
 "17:15": {
  "right": "log_jungle",
  "back": "log_jungle",
  "top": "log_jungle",
  "front": "log_jungle",
  "bottom": "log_jungle",
  "left": "log_jungle"
 },
 "62:5": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_on"
 },
 "255:2": {
  "right": "structure_block_corner",
  "back": "structure_block_corner",
  "top": "structure_block_corner",
  "front": "structure_block_corner",
  "bottom": "structure_block_corner",
  "left": "structure_block_corner"
 },
 "35:2": {
  "right": "wool_colored_magenta",
  "back": "wool_colored_magenta",
  "top": "wool_colored_magenta",
  "front": "wool_colored_magenta",
  "bottom": "wool_colored_magenta",
  "left": "wool_colored_magenta"
 },
 "125:1": {
  "right": "planks_spruce",
  "back": "planks_spruce",
  "top": "planks_spruce",
  "front": "planks_spruce",
  "bottom": "planks_spruce",
  "left": "planks_spruce"
 },
 "210:12": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_conditional",
  "top": "repeating_command_block_conditional",
  "front": "repeating_command_block_conditional",
  "bottom": "repeating_command_block_conditional",
  "left": "repeating_command_block_front"
 },
 "17:14": {
  "right": "log_birch",
  "back": "log_birch",
  "top": "log_birch",
  "front": "log_birch",
  "bottom": "log_birch",
  "left": "log_birch"
 },
 "97:0": {
  "right": "stone",
  "back": "stone",
  "top": "stone",
  "front": "stone",
  "bottom": "stone",
  "left": "stone"
 },
 "23:2": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "95:15": {
  "right": "glass_black",
  "back": "glass_black",
  "top": "glass_black",
  "front": "glass_black",
  "bottom": "glass_black",
  "left": "glass_black"
 },
 "22:0": {
  "right": "lapis_block",
  "back": "lapis_block",
  "top": "lapis_block",
  "front": "lapis_block",
  "bottom": "lapis_block",
  "left": "lapis_block"
 },
 "103:0": {
  "right": "melon_side",
  "back": "melon_side",
  "top": "melon_top",
  "front": "melon_side",
  "bottom": "melon_top",
  "left": "melon_side"
 },
 "23:3": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "43:14": {
  "right": "nether_brick",
  "back": "nether_brick",
  "top": "nether_brick",
  "front": "nether_brick",
  "bottom": "nether_brick",
  "left": "nether_brick"
 },
 "162:9": {
  "right": "log_big_oak",
  "back": "log_big_oak",
  "top": "log_big_oak_top",
  "front": "log_big_oak",
  "bottom": "log_big_oak_top",
  "left": "log_big_oak"
 },
 "62:4": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_on"
 },
 "49:0": {
  "right": "obsidian",
  "back": "obsidian",
  "top": "obsidian",
  "front": "obsidian",
  "bottom": "obsidian",
  "left": "obsidian"
 },
 "18:2": {
  "right": "leaves_birch",
  "back": "leaves_birch",
  "top": "leaves_birch",
  "front": "leaves_birch",
  "bottom": "leaves_birch",
  "left": "leaves_birch"
 },
 "211:11": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_conditional",
  "top": "chain_command_block_conditional",
  "front": "chain_command_block_conditional",
  "bottom": "chain_command_block_conditional",
  "left": "chain_command_block_front"
 },
 "155:3": {
  "right": "quartz_block_lines",
  "back": "quartz_block_lines",
  "top": "quartz_block_lines_top",
  "front": "quartz_block_lines",
  "bottom": "quartz_block_lines_top",
  "left": "quartz_block_lines"
 },
 "100:4": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_skin_red",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "43:15": {
  "right": "quartz_block_top",
  "back": "quartz_block_top",
  "top": "quartz_block_top",
  "front": "quartz_block_top",
  "bottom": "quartz_block_top",
  "left": "quartz_block_top"
 },
 "137:10": {
  "right": "command_block_back",
  "back": "command_block_conditional",
  "top": "command_block_conditional",
  "front": "command_block_conditional",
  "bottom": "command_block_conditional",
  "left": "command_block_front"
 },
 "168:2": {
  "right": "prismarine_dark",
  "back": "prismarine_dark",
  "top": "prismarine_dark",
  "front": "prismarine_dark",
  "bottom": "prismarine_dark",
  "left": "prismarine_dark"
 },
 "1:2": {
  "right": "stone_granite_smooth",
  "back": "stone_granite_smooth",
  "top": "stone_granite_smooth",
  "front": "stone_granite_smooth",
  "bottom": "stone_granite_smooth",
  "left": "stone_granite_smooth"
 },
 "137:4": {
  "right": "command_block_back",
  "back": "command_block_side",
  "top": "command_block_side",
  "front": "command_block_side",
  "bottom": "command_block_side",
  "left": "command_block_front"
 },
 "211:2": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_side",
  "top": "chain_command_block_side",
  "front": "chain_command_block_side",
  "bottom": "chain_command_block_side",
  "left": "chain_command_block_front"
 },
 "159:0": {
  "right": "hardened_clay_stained_white",
  "back": "hardened_clay_stained_white",
  "top": "hardened_clay_stained_white",
  "front": "hardened_clay_stained_white",
  "bottom": "hardened_clay_stained_white",
  "left": "hardened_clay_stained_white"
 },
 "33:1": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_normal"
 },
 "210:13": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_conditional",
  "top": "repeating_command_block_conditional",
  "front": "repeating_command_block_conditional",
  "bottom": "repeating_command_block_conditional",
  "left": "repeating_command_block_front"
 },
 "35:1": {
  "right": "wool_colored_orange",
  "back": "wool_colored_orange",
  "top": "wool_colored_orange",
  "front": "wool_colored_orange",
  "bottom": "wool_colored_orange",
  "left": "wool_colored_orange"
 },
 "95:6": {
  "right": "glass_pink",
  "back": "glass_pink",
  "top": "glass_pink",
  "front": "glass_pink",
  "bottom": "glass_pink",
  "left": "glass_pink"
 },
 "43:0": {
  "right": "stone_slab_side",
  "back": "stone_slab_side",
  "top": "stone_slab_top",
  "front": "stone_slab_side",
  "bottom": "stone_slab_top",
  "left": "stone_slab_side"
 },
 "35:13": {
  "right": "wool_colored_green",
  "back": "wool_colored_green",
  "top": "wool_colored_green",
  "front": "wool_colored_green",
  "bottom": "wool_colored_green",
  "left": "wool_colored_green"
 },
 "159:11": {
  "right": "hardened_clay_stained_blue",
  "back": "hardened_clay_stained_blue",
  "top": "hardened_clay_stained_blue",
  "front": "hardened_clay_stained_blue",
  "bottom": "hardened_clay_stained_blue",
  "left": "hardened_clay_stained_blue"
 },
 "210:0": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_side",
  "top": "repeating_command_block_side",
  "front": "repeating_command_block_side",
  "bottom": "repeating_command_block_side",
  "left": "repeating_command_block_front"
 },
 "100:14": {
  "right": "mushroom_block_skin_red",
  "back": "mushroom_block_skin_red",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_skin_red",
  "bottom": "mushroom_block_skin_red",
  "left": "mushroom_block_skin_red"
 },
 "97:4": {
  "right": "stonebrick_cracked",
  "back": "stonebrick_cracked",
  "top": "stonebrick_cracked",
  "front": "stonebrick_cracked",
  "bottom": "stonebrick_cracked",
  "left": "stonebrick_cracked"
 },
 "98:3": {
  "right": "stonebrick_carved",
  "back": "stonebrick_carved",
  "top": "stonebrick_carved",
  "front": "stonebrick_carved",
  "bottom": "stonebrick_carved",
  "left": "stonebrick_carved"
 },
 "17:1": {
  "right": "log_spruce",
  "back": "log_spruce",
  "top": "log_spruce_top",
  "front": "log_spruce",
  "bottom": "log_spruce_top",
  "left": "log_spruce"
 },
 "41:0": {
  "right": "gold_block",
  "back": "gold_block",
  "top": "gold_block",
  "front": "gold_block",
  "bottom": "gold_block",
  "left": "gold_block"
 },
 "155:4": {
  "right": "quartz_block_lines",
  "back": "quartz_block_lines",
  "top": "quartz_block_lines_top",
  "front": "quartz_block_lines",
  "bottom": "quartz_block_lines_top",
  "left": "quartz_block_lines"
 },
 "33:5": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_normal"
 },
 "25:0": {
  "right": "noteblock",
  "back": "noteblock",
  "top": "noteblock",
  "front": "noteblock",
  "bottom": "noteblock",
  "left": "noteblock"
 },
 "7:0": {
  "right": "bedrock",
  "back": "bedrock",
  "top": "bedrock",
  "front": "bedrock",
  "bottom": "bedrock",
  "left": "bedrock"
 },
 "123:0": {
  "right": "redstone_lamp_off",
  "back": "redstone_lamp_off",
  "top": "redstone_lamp_off",
  "front": "redstone_lamp_off",
  "bottom": "redstone_lamp_off",
  "left": "redstone_lamp_off"
 },
 "95:9": {
  "right": "glass_cyan",
  "back": "glass_cyan",
  "top": "glass_cyan",
  "front": "glass_cyan",
  "bottom": "glass_cyan",
  "left": "glass_cyan"
 },
 "161:5": {
  "right": "leaves_big_oak",
  "back": "leaves_big_oak",
  "top": "leaves_big_oak",
  "front": "leaves_big_oak",
  "bottom": "leaves_big_oak",
  "left": "leaves_big_oak"
 },
 "3:0": {
  "right": "dirt",
  "back": "dirt",
  "top": "dirt",
  "front": "dirt",
  "bottom": "dirt",
  "left": "dirt"
 },
 "161:9": {
  "right": "leaves_big_oak",
  "back": "leaves_big_oak",
  "top": "leaves_big_oak",
  "front": "leaves_big_oak",
  "bottom": "leaves_big_oak",
  "left": "leaves_big_oak"
 },
 "61:0": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_off"
 },
 "211:5": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_side",
  "top": "chain_command_block_side",
  "front": "chain_command_block_side",
  "bottom": "chain_command_block_side",
  "left": "chain_command_block_front"
 },
 "99:15": {
  "right": "mushroom_block_skin_stem",
  "back": "mushroom_block_skin_stem",
  "top": "mushroom_block_skin_stem",
  "front": "mushroom_block_skin_stem",
  "bottom": "mushroom_block_skin_stem",
  "left": "mushroom_block_skin_stem"
 },
 "95:5": {
  "right": "glass_lime",
  "back": "glass_lime",
  "top": "glass_lime",
  "front": "glass_lime",
  "bottom": "glass_lime",
  "left": "glass_lime"
 },
 "5:4": {
  "right": "planks_acacia",
  "back": "planks_acacia",
  "top": "planks_acacia",
  "front": "planks_acacia",
  "bottom": "planks_acacia",
  "left": "planks_acacia"
 },
 "161:0": {
  "right": "leaves_acacia",
  "back": "leaves_acacia",
  "top": "leaves_acacia",
  "front": "leaves_acacia",
  "bottom": "leaves_acacia",
  "left": "leaves_acacia"
 },
 "97:3": {
  "right": "stonebrick_mossy",
  "back": "stonebrick_mossy",
  "top": "stonebrick_mossy",
  "front": "stonebrick_mossy",
  "bottom": "stonebrick_mossy",
  "left": "stonebrick_mossy"
 },
 "159:1": {
  "right": "hardened_clay_stained_orange",
  "back": "hardened_clay_stained_orange",
  "top": "hardened_clay_stained_orange",
  "front": "hardened_clay_stained_orange",
  "bottom": "hardened_clay_stained_orange",
  "left": "hardened_clay_stained_orange"
 },
 "158:3": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "202:0": {
  "right": "purpur_pillar",
  "back": "purpur_pillar",
  "top": "purpur_pillar_top",
  "front": "purpur_pillar",
  "bottom": "purpur_pillar_top",
  "left": "purpur_pillar"
 },
 "86:0": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_off"
 },
 "62:3": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "furnace_front_on"
 },
 "95:7": {
  "right": "glass_gray",
  "back": "glass_gray",
  "top": "glass_gray",
  "front": "glass_gray",
  "bottom": "glass_gray",
  "left": "glass_gray"
 },
 "161:4": {
  "right": "leaves_acacia",
  "back": "leaves_acacia",
  "top": "leaves_acacia",
  "front": "leaves_acacia",
  "bottom": "leaves_acacia",
  "left": "leaves_acacia"
 },
 "87:0": {
  "right": "netherrack",
  "back": "netherrack",
  "top": "netherrack",
  "front": "netherrack",
  "bottom": "netherrack",
  "left": "netherrack"
 },
 "162:1": {
  "right": "log_big_oak",
  "back": "log_big_oak",
  "top": "log_big_oak_top",
  "front": "log_big_oak",
  "bottom": "log_big_oak_top",
  "left": "log_big_oak"
 },
 "88:0": {
  "right": "soul_sand",
  "back": "soul_sand",
  "top": "soul_sand",
  "front": "soul_sand",
  "bottom": "soul_sand",
  "left": "soul_sand"
 },
 "210:3": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_side",
  "top": "repeating_command_block_side",
  "front": "repeating_command_block_side",
  "bottom": "repeating_command_block_side",
  "left": "repeating_command_block_front"
 },
 "210:10": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_conditional",
  "top": "repeating_command_block_conditional",
  "front": "repeating_command_block_conditional",
  "bottom": "repeating_command_block_conditional",
  "left": "repeating_command_block_front"
 },
 "43:12": {
  "right": "brick",
  "back": "brick",
  "top": "brick",
  "front": "brick",
  "bottom": "brick",
  "left": "brick"
 },
 "29:4": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_sticky"
 },
 "211:0": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_side",
  "top": "chain_command_block_side",
  "front": "chain_command_block_side",
  "bottom": "chain_command_block_side",
  "left": "chain_command_block_front"
 },
 "211:9": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_conditional",
  "top": "chain_command_block_conditional",
  "front": "chain_command_block_conditional",
  "bottom": "chain_command_block_conditional",
  "left": "chain_command_block_front"
 },
 "97:2": {
  "right": "stonebrick",
  "back": "stonebrick",
  "top": "stonebrick",
  "front": "stonebrick",
  "bottom": "stonebrick",
  "left": "stonebrick"
 },
 "137:8": {
  "right": "command_block_back",
  "back": "command_block_conditional",
  "top": "command_block_conditional",
  "front": "command_block_conditional",
  "bottom": "command_block_conditional",
  "left": "command_block_front"
 },
 "100:8": {
  "right": "mushroom_block_skin_red",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "86:1": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_off"
 },
 "158:4": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dropper_front_horizontal"
 },
 "170:4": {
  "right": "hay_block_side",
  "back": "hay_block_side",
  "top": "hay_block_top",
  "front": "hay_block_side",
  "bottom": "hay_block_top",
  "left": "hay_block_side"
 },
 "35:14": {
  "right": "wool_colored_red",
  "back": "wool_colored_red",
  "top": "wool_colored_red",
  "front": "wool_colored_red",
  "bottom": "wool_colored_red",
  "left": "wool_colored_red"
 },
 "161:8": {
  "right": "leaves_acacia",
  "back": "leaves_acacia",
  "top": "leaves_acacia",
  "front": "leaves_acacia",
  "bottom": "leaves_acacia",
  "left": "leaves_acacia"
 },
 "210:5": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_side",
  "top": "repeating_command_block_side",
  "front": "repeating_command_block_side",
  "bottom": "repeating_command_block_side",
  "left": "repeating_command_block_front"
 },
 "17:12": {
  "right": "log_oak",
  "back": "log_oak",
  "top": "log_oak",
  "front": "log_oak",
  "bottom": "log_oak",
  "left": "log_oak"
 },
 "18:13": {
  "right": "leaves_spruce",
  "back": "leaves_spruce",
  "top": "leaves_spruce",
  "front": "leaves_spruce",
  "bottom": "leaves_spruce",
  "left": "leaves_spruce"
 },
 "33:4": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_normal"
 },
 "137:2": {
  "right": "command_block_back",
  "back": "command_block_side",
  "top": "command_block_side",
  "front": "command_block_side",
  "bottom": "command_block_side",
  "left": "command_block_front"
 },
 "48:0": {
  "right": "cobblestone_mossy",
  "back": "cobblestone_mossy",
  "top": "cobblestone_mossy",
  "front": "cobblestone_mossy",
  "bottom": "cobblestone_mossy",
  "left": "cobblestone_mossy"
 },
 "99:5": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "181:8": {
  "right": "red_sandstone_top",
  "back": "red_sandstone_top",
  "top": "red_sandstone_top",
  "front": "red_sandstone_top",
  "bottom": "red_sandstone_top",
  "left": "red_sandstone_top"
 },
 "95:3": {
  "right": "glass_light_blue",
  "back": "glass_light_blue",
  "top": "glass_light_blue",
  "front": "glass_light_blue",
  "bottom": "glass_light_blue",
  "left": "glass_light_blue"
 },
 "100:5": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "15:0": {
  "right": "iron_ore",
  "back": "iron_ore",
  "top": "iron_ore",
  "front": "iron_ore",
  "bottom": "iron_ore",
  "left": "iron_ore"
 },
 "100:2": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_red"
 },
 "137:11": {
  "right": "command_block_back",
  "back": "command_block_conditional",
  "top": "command_block_conditional",
  "front": "command_block_conditional",
  "bottom": "command_block_conditional",
  "left": "command_block_front"
 },
 "35:9": {
  "right": "wool_colored_cyan",
  "back": "wool_colored_cyan",
  "top": "wool_colored_cyan",
  "front": "wool_colored_cyan",
  "bottom": "wool_colored_cyan",
  "left": "wool_colored_cyan"
 },
 "18:3": {
  "right": "leaves_jungle",
  "back": "leaves_jungle",
  "top": "leaves_jungle",
  "front": "leaves_jungle",
  "bottom": "leaves_jungle",
  "left": "leaves_jungle"
 },
 "181:0": {
  "right": "red_sandstone_normal",
  "back": "red_sandstone_normal",
  "top": "red_sandstone_top",
  "front": "red_sandstone_normal",
  "bottom": "red_sandstone_bottom",
  "left": "red_sandstone_normal"
 },
 "211:13": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_conditional",
  "top": "chain_command_block_conditional",
  "front": "chain_command_block_conditional",
  "bottom": "chain_command_block_conditional",
  "left": "chain_command_block_front"
 },
 "17:4": {
  "right": "log_oak",
  "back": "log_oak",
  "top": "log_oak_top",
  "front": "log_oak",
  "bottom": "log_oak_top",
  "left": "log_oak"
 },
 "158:1": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dropper_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "159:10": {
  "right": "hardened_clay_stained_purple",
  "back": "hardened_clay_stained_purple",
  "top": "hardened_clay_stained_purple",
  "front": "hardened_clay_stained_purple",
  "bottom": "hardened_clay_stained_purple",
  "left": "hardened_clay_stained_purple"
 },
 "137:9": {
  "right": "command_block_back",
  "back": "command_block_conditional",
  "top": "command_block_conditional",
  "front": "command_block_conditional",
  "bottom": "command_block_conditional",
  "left": "command_block_front"
 },
 "35:6": {
  "right": "wool_colored_pink",
  "back": "wool_colored_pink",
  "top": "wool_colored_pink",
  "front": "wool_colored_pink",
  "bottom": "wool_colored_pink",
  "left": "wool_colored_pink"
 },
 "17:2": {
  "right": "log_birch",
  "back": "log_birch",
  "top": "log_birch_top",
  "front": "log_birch",
  "bottom": "log_birch_top",
  "left": "log_birch"
 },
 "43:11": {
  "right": "cobblestone",
  "back": "cobblestone",
  "top": "cobblestone",
  "front": "cobblestone",
  "bottom": "cobblestone",
  "left": "cobblestone"
 },
 "95:10": {
  "right": "glass_purple",
  "back": "glass_purple",
  "top": "glass_purple",
  "front": "glass_purple",
  "bottom": "glass_purple",
  "left": "glass_purple"
 },
 "12:1": {
  "right": "red_sand",
  "back": "red_sand",
  "top": "red_sand",
  "front": "red_sand",
  "bottom": "red_sand",
  "left": "red_sand"
 },
 "98:2": {
  "right": "stonebrick_cracked",
  "back": "stonebrick_cracked",
  "top": "stonebrick_cracked",
  "front": "stonebrick_cracked",
  "bottom": "stonebrick_cracked",
  "left": "stonebrick_cracked"
 },
 "46:0": {
  "right": "tnt_side",
  "back": "tnt_side",
  "top": "tnt_top",
  "front": "tnt_side",
  "bottom": "tnt_bottom",
  "left": "tnt_side"
 },
 "159:6": {
  "right": "hardened_clay_stained_pink",
  "back": "hardened_clay_stained_pink",
  "top": "hardened_clay_stained_pink",
  "front": "hardened_clay_stained_pink",
  "bottom": "hardened_clay_stained_pink",
  "left": "hardened_clay_stained_pink"
 },
 "179:1": {
  "right": "red_sandstone_carved",
  "back": "red_sandstone_carved",
  "top": "red_sandstone_top",
  "front": "red_sandstone_carved",
  "bottom": "red_sandstone_top",
  "left": "red_sandstone_carved"
 },
 "99:10": {
  "right": "mushroom_block_skin_stem",
  "back": "mushroom_block_skin_stem",
  "top": "mushroom_block_inside",
  "front": "mushroom_block_skin_stem",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_stem"
 },
 "211:1": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_side",
  "top": "chain_command_block_side",
  "front": "chain_command_block_side",
  "bottom": "chain_command_block_side",
  "left": "chain_command_block_front"
 },
 "159:4": {
  "right": "hardened_clay_stained_yellow",
  "back": "hardened_clay_stained_yellow",
  "top": "hardened_clay_stained_yellow",
  "front": "hardened_clay_stained_yellow",
  "bottom": "hardened_clay_stained_yellow",
  "left": "hardened_clay_stained_yellow"
 },
 "17:9": {
  "right": "log_spruce",
  "back": "log_spruce",
  "top": "log_spruce_top",
  "front": "log_spruce",
  "bottom": "log_spruce_top",
  "left": "log_spruce"
 },
 "43:10": {
  "right": "planks_oak",
  "back": "planks_oak",
  "top": "planks_oak",
  "front": "planks_oak",
  "bottom": "planks_oak",
  "left": "planks_oak"
 },
 "155:2": {
  "right": "quartz_block_lines",
  "back": "quartz_block_lines",
  "top": "quartz_block_lines_top",
  "front": "quartz_block_lines",
  "bottom": "quartz_block_lines_top",
  "left": "quartz_block_lines"
 },
 "29:2": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_sticky"
 },
 "202:4": {
  "right": "purpur_pillar",
  "back": "purpur_pillar",
  "top": "purpur_pillar_top",
  "front": "purpur_pillar",
  "bottom": "purpur_pillar_top",
  "left": "purpur_pillar"
 },
 "255:1": {
  "right": "structure_block_load",
  "back": "structure_block_load",
  "top": "structure_block_load",
  "front": "structure_block_load",
  "bottom": "structure_block_load",
  "left": "structure_block_load"
 },
 "162:8": {
  "right": "log_acacia",
  "back": "log_acacia",
  "top": "log_acacia_top",
  "front": "log_acacia",
  "bottom": "log_acacia_top",
  "left": "log_acacia"
 },
 "35:5": {
  "right": "wool_colored_lime",
  "back": "wool_colored_lime",
  "top": "wool_colored_lime",
  "front": "wool_colored_lime",
  "bottom": "wool_colored_lime",
  "left": "wool_colored_lime"
 },
 "23:12": {
  "right": "furnace_side",
  "back": "furnace_side",
  "top": "furnace_top",
  "front": "furnace_side",
  "bottom": "furnace_top",
  "left": "dispenser_front_horizontal"
 },
 "95:14": {
  "right": "glass_red",
  "back": "glass_red",
  "top": "glass_red",
  "front": "glass_red",
  "bottom": "glass_red",
  "left": "glass_red"
 },
 "1:5": {
  "right": "stone_andesite",
  "back": "stone_andesite",
  "top": "stone_andesite",
  "front": "stone_andesite",
  "bottom": "stone_andesite",
  "left": "stone_andesite"
 },
 "95:12": {
  "right": "glass_brown",
  "back": "glass_brown",
  "top": "glass_brown",
  "front": "glass_brown",
  "bottom": "glass_brown",
  "left": "glass_brown"
 },
 "1:1": {
  "right": "stone_granite",
  "back": "stone_granite",
  "top": "stone_granite",
  "front": "stone_granite",
  "bottom": "stone_granite",
  "left": "stone_granite"
 },
 "35:12": {
  "right": "wool_colored_brown",
  "back": "wool_colored_brown",
  "top": "wool_colored_brown",
  "front": "wool_colored_brown",
  "bottom": "wool_colored_brown",
  "left": "wool_colored_brown"
 },
 "91:1": {
  "right": "pumpkin_side",
  "back": "pumpkin_side",
  "top": "pumpkin_top",
  "front": "pumpkin_side",
  "bottom": "pumpkin_top",
  "left": "pumpkin_face_on"
 },
 "99:6": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_skin_brown",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "43:5": {
  "right": "stonebrick",
  "back": "stonebrick",
  "top": "stonebrick",
  "front": "stonebrick",
  "bottom": "stonebrick",
  "left": "stonebrick"
 },
 "78:7": {
  "right": "snow",
  "back": "snow",
  "top": "snow",
  "front": "snow",
  "bottom": "snow",
  "left": "snow"
 },
 "161:1": {
  "right": "leaves_big_oak",
  "back": "leaves_big_oak",
  "top": "leaves_big_oak",
  "front": "leaves_big_oak",
  "bottom": "leaves_big_oak",
  "left": "leaves_big_oak"
 },
 "29:1": {
  "right": "piston_bottom",
  "back": "piston_side",
  "top": "piston_side",
  "front": "piston_side",
  "bottom": "piston_side",
  "left": "piston_top_sticky"
 },
 "168:1": {
  "right": "prismarine_bricks",
  "back": "prismarine_bricks",
  "top": "prismarine_bricks",
  "front": "prismarine_bricks",
  "bottom": "prismarine_bricks",
  "left": "prismarine_bricks"
 },
 "74:0": {
  "right": "redstone_ore",
  "back": "redstone_ore",
  "top": "redstone_ore",
  "front": "redstone_ore",
  "bottom": "redstone_ore",
  "left": "redstone_ore"
 },
 "137:13": {
  "right": "command_block_back",
  "back": "command_block_conditional",
  "top": "command_block_conditional",
  "front": "command_block_conditional",
  "bottom": "command_block_conditional",
  "left": "command_block_front"
 },
 "99:8": {
  "right": "mushroom_block_skin_brown",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_inside"
 },
 "162:0": {
  "right": "log_acacia",
  "back": "log_acacia",
  "top": "log_acacia_top",
  "front": "log_acacia",
  "bottom": "log_acacia_top",
  "left": "log_acacia"
 },
 "43:1": {
  "right": "sandstone_normal",
  "back": "sandstone_normal",
  "top": "sandstone_top",
  "front": "sandstone_normal",
  "bottom": "sandstone_bottom",
  "left": "sandstone_normal"
 },
 "23:8": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dispenser_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "159:7": {
  "right": "hardened_clay_stained_gray",
  "back": "hardened_clay_stained_gray",
  "top": "hardened_clay_stained_gray",
  "front": "hardened_clay_stained_gray",
  "bottom": "hardened_clay_stained_gray",
  "left": "hardened_clay_stained_gray"
 },
 "125:0": {
  "right": "planks_oak",
  "back": "planks_oak",
  "top": "planks_oak",
  "front": "planks_oak",
  "bottom": "planks_oak",
  "left": "planks_oak"
 },
 "137:1": {
  "right": "command_block_back",
  "back": "command_block_side",
  "top": "command_block_side",
  "front": "command_block_side",
  "bottom": "command_block_side",
  "left": "command_block_front"
 },
 "162:5": {
  "right": "log_big_oak",
  "back": "log_big_oak",
  "top": "log_big_oak_top",
  "front": "log_big_oak",
  "bottom": "log_big_oak_top",
  "left": "log_big_oak"
 },
 "255:3": {
  "right": "structure_block_data",
  "back": "structure_block_data",
  "top": "structure_block_data",
  "front": "structure_block_data",
  "bottom": "structure_block_data",
  "left": "structure_block_data"
 },
 "210:2": {
  "right": "repeating_command_block_back",
  "back": "repeating_command_block_side",
  "top": "repeating_command_block_side",
  "front": "repeating_command_block_side",
  "bottom": "repeating_command_block_side",
  "left": "repeating_command_block_front"
 },
 "5:1": {
  "right": "planks_spruce",
  "back": "planks_spruce",
  "top": "planks_spruce",
  "front": "planks_spruce",
  "bottom": "planks_spruce",
  "left": "planks_spruce"
 },
 "201:0": {
  "right": "purpur_block",
  "back": "purpur_block",
  "top": "purpur_block",
  "front": "purpur_block",
  "bottom": "purpur_block",
  "left": "purpur_block"
 },
 "17:0": {
  "right": "log_oak",
  "back": "log_oak",
  "top": "log_oak_top",
  "front": "log_oak",
  "bottom": "log_oak_top",
  "left": "log_oak"
 },
 "1:6": {
  "right": "stone_andesite_smooth",
  "back": "stone_andesite_smooth",
  "top": "stone_andesite_smooth",
  "front": "stone_andesite_smooth",
  "bottom": "stone_andesite_smooth",
  "left": "stone_andesite_smooth"
 },
 "5:3": {
  "right": "planks_jungle",
  "back": "planks_jungle",
  "top": "planks_jungle",
  "front": "planks_jungle",
  "bottom": "planks_jungle",
  "left": "planks_jungle"
 },
 "42:0": {
  "right": "iron_block",
  "back": "iron_block",
  "top": "iron_block",
  "front": "iron_block",
  "bottom": "iron_block",
  "left": "iron_block"
 },
 "99:2": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_brown",
  "front": "mushroom_block_inside",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_brown"
 },
 "100:1": {
  "right": "mushroom_block_inside",
  "back": "mushroom_block_inside",
  "top": "mushroom_block_skin_red",
  "front": "mushroom_block_skin_red",
  "bottom": "mushroom_block_inside",
  "left": "mushroom_block_skin_red"
 },
 "211:4": {
  "right": "chain_command_block_back",
  "back": "chain_command_block_side",
  "top": "chain_command_block_side",
  "front": "chain_command_block_side",
  "bottom": "chain_command_block_side",
  "left": "chain_command_block_front"
 },
 "159:13": {
  "right": "hardened_clay_stained_green",
  "back": "hardened_clay_stained_green",
  "top": "hardened_clay_stained_green",
  "front": "hardened_clay_stained_green",
  "bottom": "hardened_clay_stained_green",
  "left": "hardened_clay_stained_green"
 },
 "159:8": {
  "right": "hardened_clay_stained_silver",
  "back": "hardened_clay_stained_silver",
  "top": "hardened_clay_stained_silver",
  "front": "hardened_clay_stained_silver",
  "bottom": "hardened_clay_stained_silver",
  "left": "hardened_clay_stained_silver"
 },
 "23:0": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dispenser_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 },
 "158:8": {
  "right": "furnace_top",
  "back": "furnace_top",
  "top": "dropper_front_vertical",
  "front": "furnace_top",
  "bottom": "furnace_top",
  "left": "furnace_top"
 }
}
// END JSON OBJECT
];

const blocks = rawData[0];

const rawDataUV = [
// BEGIN JSON OBJECT
{
 "187:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 90,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 90,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "131:14": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      6,
      16,
      8
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      8,
      16,
      6
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    0.5,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     0,
     0
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    8.25,
    0.5,
    6.7
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    3.4,
    6.7
   ],
   "to": [
    9.8,
    4.2,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    9.1
   ],
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "to": [
    8.6,
    4.2,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "to": [
    7.4,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    3.4,
    7.9
   ],
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "145:2": [
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    2,
    0,
    2
   ],
   "yRot": 180,
   "to": [
    14,
    3,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "anvil_side"
    },
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    3,
    3
   ],
   "yRot": 180,
   "to": [
    13,
    5,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_top_damaged_0"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    11,
    0
   ],
   "yRot": 180,
   "to": [
    13,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    5,
    5,
    4
   ],
   "yRot": 180,
   "to": [
    11,
    10,
    12
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    },
    "down": {
     "texture": "anvil_base"
    }
   },
   "from": [
    4,
    10,
    1
   ],
   "yRot": 180,
   "to": [
    12,
    12,
    15
   ]
  }
 ],
 "72:0": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      16
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      16
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      16
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    1,
    15
   ]
  }
 ],
 "185:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "185:12": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "160:3": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_light_blue"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_light_blue"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "149:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    4,
    2,
    11
   ],
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    10,
    2,
    11
   ],
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    5,
    2
   ],
   "to": [
    9,
    5,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "to": [
    9,
    6,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "to": [
    10,
    6,
    4
   ]
  }
 ],
 "140:0": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      10,
      6,
      16
     ],
     "texture": "flower_pot"
    },
    "up": {
     "uv": [
      5,
      5,
      6,
      11
     ],
     "texture": "flower_pot"
    },
    "down": {
     "uv": [
      5,
      5,
      6,
      11
     ],
     "cullface": "down",
     "texture": "flower_pot"
    },
    "north": {
     "uv": [
      10,
      10,
      11,
      16
     ],
     "texture": "flower_pot"
    },
    "east": {
     "uv": [
      5,
      10,
      11,
      16
     ],
     "texture": "flower_pot"
    },
    "west": {
     "uv": [
      5,
      10,
      11,
      16
     ],
     "texture": "flower_pot"
    }
   },
   "from": [
    5,
    0,
    5
   ],
   "to": [
    6,
    6,
    11
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      10,
      10,
      11,
      16
     ],
     "texture": "flower_pot"
    },
    "up": {
     "uv": [
      10,
      5,
      11,
      11
     ],
     "texture": "flower_pot"
    },
    "down": {
     "uv": [
      10,
      5,
      11,
      11
     ],
     "cullface": "down",
     "texture": "flower_pot"
    },
    "north": {
     "uv": [
      5,
      10,
      6,
      16
     ],
     "texture": "flower_pot"
    },
    "east": {
     "uv": [
      5,
      10,
      11,
      16
     ],
     "texture": "flower_pot"
    },
    "west": {
     "uv": [
      5,
      10,
      11,
      16
     ],
     "texture": "flower_pot"
    }
   },
   "from": [
    10,
    0,
    5
   ],
   "to": [
    11,
    6,
    11
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      10,
      10,
      16
     ],
     "texture": "flower_pot"
    },
    "south": {
     "uv": [
      6,
      10,
      10,
      16
     ],
     "texture": "flower_pot"
    },
    "up": {
     "uv": [
      6,
      5,
      10,
      6
     ],
     "texture": "flower_pot"
    },
    "down": {
     "uv": [
      6,
      10,
      10,
      11
     ],
     "cullface": "down",
     "texture": "flower_pot"
    }
   },
   "from": [
    6,
    0,
    5
   ],
   "to": [
    10,
    6,
    6
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      10,
      10,
      16
     ],
     "texture": "flower_pot"
    },
    "south": {
     "uv": [
      6,
      10,
      10,
      16
     ],
     "texture": "flower_pot"
    },
    "up": {
     "uv": [
      6,
      10,
      10,
      11
     ],
     "texture": "flower_pot"
    },
    "down": {
     "uv": [
      6,
      5,
      10,
      6
     ],
     "cullface": "down",
     "texture": "flower_pot"
    }
   },
   "from": [
    6,
    0,
    10
   ],
   "to": [
    10,
    6,
    11
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "texture": "dirt"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "flower_pot"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "to": [
    10,
    4,
    10
   ]
  }
 ],
 "60:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "dirt"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "farmland_dry"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "dirt"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "dirt"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "dirt"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "66:6": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "down": {
     "uv": [
      0,
      16,
      16,
      0
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      2,
      15,
      3
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    11,
    0,
    2
   ],
   "to": [
    16,
    1,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    2,
    0,
    11
   ],
   "to": [
    4,
    1,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      12,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    3,
    0,
    9
   ],
   "to": [
    5,
    1,
    12
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      12,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    9,
    0,
    3
   ],
   "to": [
    12,
    1,
    5
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    4,
    0,
    7
   ],
   "to": [
    6,
    1,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    7,
    0,
    4
   ],
   "to": [
    10,
    1,
    6
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      3,
      14,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "to": [
    7,
    1,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    6,
    0,
    5
   ],
   "to": [
    8,
    1,
    7
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    12,
    0,
    14
   ],
   "to": [
    14,
    1,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    14,
    0,
    12
   ],
   "to": [
    16,
    1,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    13,
    0,
    13
   ],
   "to": [
    15,
    1,
    15
   ]
  }
 ],
 "93:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    6
   ],
   "to": [
    9,
    7,
    8
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "to": [
    9,
    7,
    4
   ]
  }
 ],
 "6:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "109:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stonebrick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "stonebrick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "178:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "33:10": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "piston_bottom"
    },
    "up": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_inner"
    },
    "east": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "east",
     "texture": "piston_side",
     "rotation": 90
    },
    "west": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "west",
     "texture": "piston_side",
     "rotation": 270
    }
   },
   "from": [
    0,
    0,
    4
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "208:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "grass_path_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "grass_path_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "grass_path_side"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "grass_path_side"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "grass_path_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "147:8": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "gold_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "gold_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "83:8": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "108:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "brick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "brick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "brick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "149:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    4,
    2,
    11
   ],
   "yRot": 90,
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    10,
    2,
    11
   ],
   "yRot": 90,
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    5,
    2
   ],
   "yRot": 90,
   "to": [
    9,
    5,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "yRot": 90,
   "to": [
    9,
    6,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "yRot": 90,
   "to": [
    10,
    6,
    4
   ]
  }
 ],
 "69:7": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 180,
   "yRot": 180,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "51:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "76:3": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_on"
    }
   },
   "shade": false,
   "to": [
    1,
    13.5,
    9
   ],
   "from": [
    -1,
    3.5,
    7
   ],
   "yRot": 90,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    }
   },
   "shade": false,
   "to": [
    1,
    19.5,
    16
   ],
   "from": [
    -1,
    3.5,
    0
   ],
   "yRot": 90,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    }
   },
   "shade": false,
   "to": [
    8,
    19.5,
    9
   ],
   "from": [
    -8,
    3.5,
    7
   ],
   "yRot": 90,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  }
 ],
 "160:11": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_blue"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_blue"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "182:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "148:3": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "81:15": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "143:13": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "to": [
    11,
    1,
    10
   ]
  }
 ],
 "28:12": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_detector_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "167:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "south",
     "texture": "iron_trapdoor"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "iron_trapdoor"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "iron_trapdoor"
    },
    "north": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "north",
     "texture": "iron_trapdoor"
    },
    "east": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "east",
     "texture": "iron_trapdoor"
    },
    "west": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "west",
     "texture": "iron_trapdoor"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    3,
    16
   ]
  }
 ],
 "50:4": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "torch_on"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "torch_on"
    }
   },
   "shade": false,
   "to": [
    1,
    13.5,
    9
   ],
   "from": [
    -1,
    3.5,
    7
   ],
   "yRot": 270,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    }
   },
   "shade": false,
   "to": [
    1,
    19.5,
    16
   ],
   "from": [
    -1,
    3.5,
    0
   ],
   "yRot": 270,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    }
   },
   "shade": false,
   "to": [
    8,
    19.5,
    9
   ],
   "from": [
    -8,
    3.5,
    7
   ],
   "yRot": 270,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  }
 ],
 "167:15": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "iron_trapdoor"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "up",
     "texture": "iron_trapdoor"
    },
    "down": {
     "uv": [
      0,
      13,
      16,
      16
     ],
     "cullface": "down",
     "texture": "iron_trapdoor"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "iron_trapdoor"
    },
    "east": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "iron_trapdoor"
    },
    "west": {
     "uv": [
      16,
      0,
      13,
      16
     ],
     "cullface": "west",
     "texture": "iron_trapdoor"
    }
   },
   "from": [
    0,
    0,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "83:1": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "184:7": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "44:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "brick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "55:14": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "142:5": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "51:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "60:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "dirt"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "farmland_dry"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "dirt"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "dirt"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "dirt"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "142:1": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_0"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "143:5": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "to": [
    11,
    2,
    10
   ]
  }
 ],
 "64:2": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_lower"
    },
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_wood_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "128:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "sandstone_normal"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "92:0": [
  {
   "faces": {
    "south": {
     "texture": "cake_side"
    },
    "up": {
     "texture": "cake_top"
    },
    "down": {
     "cullface": "down",
     "texture": "cake_bottom"
    },
    "north": {
     "texture": "cake_side"
    },
    "east": {
     "texture": "cake_side"
    },
    "west": {
     "texture": "cake_side"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    8,
    15
   ]
  }
 ],
 "151:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "59:0": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_0"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "178:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "198:1": [
  {
   "faces": {
    "south": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      2,
      6,
      6
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      6,
      6,
      2,
      2
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "to": [
    10,
    1,
    10
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      2
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      4,
      2,
      2,
      0
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    7,
    1,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "96:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "south",
     "texture": "trapdoor"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "trapdoor"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "trapdoor"
    },
    "north": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "north",
     "texture": "trapdoor"
    },
    "east": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "east",
     "texture": "trapdoor"
    },
    "west": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "west",
     "texture": "trapdoor"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    3,
    16
   ]
  }
 ],
 "142:4": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "196:0": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      16.0
     ],
     "cullface": "north",
     "texture": "door_acacia_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      2.0,
      0.0,
      0.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      2.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    3.0,
    16.0,
    2.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_acacia_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      14.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      14.0,
      0.0,
      16.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    14.0
   ],
   "name": "Element",
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      8.0,
      2.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_side"
    },
    "down": {
     "uv": [
      8.0,
      14.0,
      11.0,
      2.0
     ],
     "cullface": "down",
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      14.0,
      14.0,
      2.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      14.0,
      14.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    2.0
   ],
   "name": "Element",
   "to": [
    3.0,
    2.0,
    14.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    2.0
   ],
   "name": "Element",
   "to": [
    2.0,
    3.0,
    3.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    13.0
   ],
   "name": "Element",
   "to": [
    2.0,
    3.0,
    14.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    5.0
   ],
   "name": "Element",
   "to": [
    2.0,
    4.0,
    6.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    10.0
   ],
   "name": "Element",
   "to": [
    2.0,
    4.0,
    11.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      11.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "up": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      4.0,
      12.0,
      5.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    4.0
   ],
   "name": "Element",
   "to": [
    2.0,
    4.0,
    12.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    5.0,
    2.0
   ],
   "name": "Element",
   "to": [
    2.0,
    6.0,
    4.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    5.0,
    12.0
   ],
   "name": "Element",
   "to": [
    2.0,
    6.0,
    14.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    9.0
   ],
   "name": "Element",
   "to": [
    2.0,
    16.0,
    10.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    6.0
   ],
   "name": "Element",
   "to": [
    2.0,
    16.0,
    7.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    3.0
   ],
   "name": "Element",
   "to": [
    2.0,
    16.0,
    4.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      12.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    12.0
   ],
   "name": "Element",
   "to": [
    2.0,
    16.0,
    13.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      11.0,
      11.0,
      12.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    11.0
   ],
   "name": "Element",
   "to": [
    2.0,
    5.0,
    13.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      4.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    3.0
   ],
   "name": "Element",
   "to": [
    2.0,
    5.0,
    5.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    13.0,
    2.0
   ],
   "name": "Element",
   "to": [
    2.0,
    14.0,
    14.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    9.0,
    2.0
   ],
   "name": "Element",
   "to": [
    2.0,
    10.0,
    14.0
   ]
  }
 ],
 "131:12": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      6,
      16,
      8
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      8,
      16,
      6
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    0.5,
    0
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     0,
     0
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    8.25,
    0.5,
    6.7
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    3.4,
    6.7
   ],
   "yRot": 180,
   "to": [
    9.8,
    4.2,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    9.1
   ],
   "yRot": 180,
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "yRot": 180,
   "to": [
    8.6,
    4.2,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "yRot": 180,
   "to": [
    7.4,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    3.4,
    7.9
   ],
   "yRot": 180,
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "yRot": 180,
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "60:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "dirt"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "farmland_dry"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "dirt"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "dirt"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "dirt"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "154:12": [
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_inside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    10,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    11,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    14,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    14
   ],
   "yRot": 270,
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    4,
    4,
    4
   ],
   "yRot": 270,
   "to": [
    12,
    10,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    6,
    4,
    0
   ],
   "yRot": 270,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "96:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "south",
     "texture": "trapdoor"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "trapdoor"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "trapdoor"
    },
    "north": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "north",
     "texture": "trapdoor"
    },
    "east": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "east",
     "texture": "trapdoor"
    },
    "west": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "west",
     "texture": "trapdoor"
    }
   },
   "from": [
    0,
    13,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "106:10": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    15.2,
    0,
    0
   ],
   "shade": false,
   "to": [
    15.2,
    16,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0.8,
    0,
    0
   ],
   "shade": false,
   "to": [
    0.8,
    16,
    16
   ]
  }
 ],
 "71:7": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_iron_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "175:8": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "texture": "double_plant_sunflower_top"
    },
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "texture": "double_plant_sunflower_top"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    8,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "texture": "double_plant_sunflower_top"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "texture": "double_plant_sunflower_top"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    8,
    15.2
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_sunflower_front"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_sunflower_back"
    }
   },
   "from": [
    9.6,
    -1,
    1
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    9.6,
    15,
    15
   ],
   "shade": false
  }
 ],
 "27:10": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "151:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "51:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "81:5": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "28:5": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_detector"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_detector"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    }
   },
   "from": [
    2,
    7.25,
    0
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16.75
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    }
   },
   "from": [
    12,
    7.25,
    0
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16.75
   ]
  }
 ],
 "81:7": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "131:9": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    4.2,
    6.7
   ],
   "yRot": 270,
   "to": [
    9.8,
    5,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    9.1
   ],
   "yRot": 270,
   "to": [
    8.6,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    7.9
   ],
   "yRot": 270,
   "to": [
    8.6,
    5,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    7.9
   ],
   "yRot": 270,
   "to": [
    7.4,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    4.2,
    7.9
   ],
   "yRot": 270,
   "to": [
    8.6,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "yRot": 270,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "yRot": 270,
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "104:4": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      10
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "south": {
     "uv": [
      16,
      0,
      0,
      10
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    0,
    -1,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    16,
    9,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      10
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      10
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    8,
    -1,
    0
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    9,
    16
   ]
  }
 ],
 "167:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "south",
     "texture": "iron_trapdoor"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "iron_trapdoor"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "iron_trapdoor"
    },
    "north": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "north",
     "texture": "iron_trapdoor"
    },
    "east": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "east",
     "texture": "iron_trapdoor"
    },
    "west": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "west",
     "texture": "iron_trapdoor"
    }
   },
   "from": [
    0,
    13,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "107:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "147:3": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "gold_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "gold_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "183:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "67:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "cobblestone"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "183:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "151:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "148:14": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "51:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "81:10": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "38:5": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_orange"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_orange"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_orange"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_orange"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "143:9": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 90,
   "to": [
    11,
    1,
    10
   ]
  }
 ],
 "53:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "58:0": [
  {
   "faces": {
    "up": {
     "uv": [
      0.5,
      0.5,
      15.5,
      15.5
     ],
     "texture": "crafting_table_top"
    },
    "north": {
     "uv": [
      0.5,
      0.0,
      15.5,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "south": {
     "uv": [
      0.5,
      0.0,
      15.5,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      0.5,
      0.0,
      15.5,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      0.5,
      0.0,
      15.5,
      16.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    0.5,
    0.0,
    0.5
   ],
   "name": "Element",
   "to": [
    15.5,
    16.0,
    15.5
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0.0,
      0.0,
      3.0,
      3.0
     ],
     "texture": "crafting_table_inside"
    },
    "north": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "south": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    13.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "to": [
    16.0,
    14.0,
    16.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "south": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    3.0,
    14.0,
    3.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "south": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    13.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    14.0,
    3.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "south": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      0.0,
      2.0,
      3.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      13.0,
      2.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "to": [
    3.0,
    14.0,
    16.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.0,
      0.0,
      16.0,
      2.0
     ],
     "texture": "crafting_table_front"
    },
    "up": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_top"
    },
    "down": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_inside"
    },
    "north": {
     "uv": [
      0.0,
      0.0,
      16.0,
      2.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      0.0,
      0.0,
      16.0,
      2.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      2.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    0.0,
    14.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "up": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_inside"
    },
    "down": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "rotation": 180,
     "texture": "crafting_table_bottom"
    },
    "north": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "east": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    },
    "west": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "crafting_table_front"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    1.0,
    16.0
   ]
  }
 ],
 "34:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "yRot": 90,
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "109:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stonebrick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "stonebrick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "185:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "150:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    4,
    2,
    11
   ],
   "yRot": 90,
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    10,
    2,
    11
   ],
   "yRot": 90,
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "yRot": 90,
   "to": [
    9,
    4,
    4
   ]
  }
 ],
 "160:6": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_pink"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_pink"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "199:0": [
  {
   "faces": {
    "north": {
     "texture": "chorus_plant"
    }
   },
   "from": [
    4,
    4,
    4
   ],
   "to": [
    12,
    12,
    12
   ]
  }
 ],
 "59:5": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_5"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "27:8": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    2,
    0,
    -0.25
   ],
   "to": [
    4,
    1,
    16.25
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    12,
    0,
    -0.25
   ],
   "to": [
    14,
    1,
    16.25
   ]
  }
 ],
 "145:6": [
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    2,
    0,
    2
   ],
   "yRot": 180,
   "to": [
    14,
    3,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "anvil_side"
    },
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    3,
    3
   ],
   "yRot": 180,
   "to": [
    13,
    5,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_top_damaged_1"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    11,
    0
   ],
   "yRot": 180,
   "to": [
    13,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    5,
    5,
    4
   ],
   "yRot": 180,
   "to": [
    11,
    10,
    12
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    },
    "down": {
     "texture": "anvil_base"
    }
   },
   "from": [
    4,
    10,
    1
   ],
   "yRot": 180,
   "to": [
    12,
    12,
    15
   ]
  }
 ],
 "148:7": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "72:1": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "94:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    12
   ],
   "yRot": 90,
   "to": [
    9,
    7,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    11
   ],
   "yRot": 90,
   "to": [
    9,
    8,
    15
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    12
   ],
   "yRot": 90,
   "to": [
    10,
    8,
    14
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    2
   ],
   "yRot": 90,
   "to": [
    9,
    7,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "yRot": 90,
   "to": [
    9,
    8,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "yRot": 90,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "71:1": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_lower"
    },
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_iron_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "148:12": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "104:3": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "south": {
     "uv": [
      16,
      0,
      0,
      8
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    0,
    -1,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    16,
    7,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      8
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    8,
    -1,
    0
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    7,
    16
   ]
  }
 ],
 "104:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      2
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "south": {
     "uv": [
      16,
      0,
      0,
      2
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    0,
    -1,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    16,
    1,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      2
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      2
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    8,
    -1,
    0
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    1,
    16
   ]
  }
 ],
 "148:13": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "107:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "160:9": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_cyan"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_cyan"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "111:0": [
  {
   "faces": {
    "up": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "waterlily",
     "tintindex": 0
    },
    "down": {
     "uv": [
      16,
      16,
      0,
      0
     ],
     "texture": "waterlily",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "175:3": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_fern_bottom",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_fern_bottom",
     "tintindex": 0
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_fern_bottom",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_fern_bottom",
     "tintindex": 0
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "50:2": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "torch_on"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "torch_on"
    }
   },
   "shade": false,
   "to": [
    1,
    13.5,
    9
   ],
   "from": [
    -1,
    3.5,
    7
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    }
   },
   "shade": false,
   "to": [
    1,
    19.5,
    16
   ],
   "from": [
    -1,
    3.5,
    0
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "torch_on"
    }
   },
   "shade": false,
   "to": [
    8,
    19.5,
    9
   ],
   "from": [
    -8,
    3.5,
    7
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   }
  }
 ],
 "55:1": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "44:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "north",
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "west",
     "texture": "cobblestone"
    }
   },
   "from": [
    0,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "44:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "151:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "106:14": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    15.2,
    0,
    0
   ],
   "yRot": 270,
   "shade": false,
   "to": [
    15.2,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    15.2
   ],
   "yRot": 270,
   "shade": false,
   "to": [
    16,
    16,
    15.2
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    0.8
   ],
   "yRot": 270,
   "shade": false,
   "to": [
    16,
    16,
    0.8
   ]
  }
 ],
 "69:2": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 90,
   "yRot": 270,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 90,
   "yRot": 270,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "180:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "141:5": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_2"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "108:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "brick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "brick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "brick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "85:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_oak"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "up",
     "texture": "fence_oak"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "fence_oak"
    },
    "north": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_oak"
    },
    "east": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_oak"
    },
    "west": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_oak"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "__comment": "Center post",
   "to": [
    10,
    16,
    10
   ]
  }
 ],
 "81:1": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "6:8": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_oak"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "150:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    4,
    2,
    11
   ],
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    10,
    2,
    11
   ],
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    5,
    2
   ],
   "to": [
    9,
    5,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "to": [
    9,
    6,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "to": [
    10,
    6,
    4
   ]
  }
 ],
 "142:6": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_2"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "70:1": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "stone"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "stone"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "stone"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "stone"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "stone"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "stone"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "131:13": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      6,
      16,
      8
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      8,
      16,
      6
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    0.5,
    0
   ],
   "yRot": 270,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     0,
     0
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    8.25,
    0.5,
    6.7
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    3.4,
    6.7
   ],
   "yRot": 270,
   "to": [
    9.8,
    4.2,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    9.1
   ],
   "yRot": 270,
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "yRot": 270,
   "to": [
    8.6,
    4.2,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "yRot": 270,
   "to": [
    7.4,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    3.4,
    7.9
   ],
   "yRot": 270,
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "yRot": 270,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "yRot": 270,
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "83:13": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "77:11": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "stone"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "stone"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "stone"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "stone"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "stone"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "stone"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 180,
   "to": [
    11,
    1,
    10
   ]
  }
 ],
 "167:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "south",
     "texture": "iron_trapdoor"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "iron_trapdoor"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "iron_trapdoor"
    },
    "north": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "north",
     "texture": "iron_trapdoor"
    },
    "east": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "east",
     "texture": "iron_trapdoor"
    },
    "west": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "west",
     "texture": "iron_trapdoor"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    3,
    16
   ]
  }
 ],
 "156:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "quartz_block_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "quartz_block_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "quartz_block_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "quartz_block_side"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "quartz_block_side"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "quartz_block_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "quartz_block_side"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "quartz_block_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "quartz_block_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "quartz_block_side"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "quartz_block_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "quartz_block_side"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "77:12": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "stone"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "stone"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "stone"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "stone"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "stone"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "stone"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "to": [
    11,
    1,
    10
   ]
  }
 ],
 "83:12": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "184:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "178:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "183:15": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "203:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "purpur_block"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "purpur_block"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "34:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "yRot": 270,
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "163:7": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_acacia"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_acacia"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "135:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_birch"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_birch"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "107:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "64:10": [
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_wood_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "171:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "south",
     "texture": "wool_colored_magenta"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wool_colored_magenta"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "wool_colored_magenta"
    },
    "north": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "north",
     "texture": "wool_colored_magenta"
    },
    "east": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "east",
     "texture": "wool_colored_magenta"
    },
    "west": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "west",
     "texture": "wool_colored_magenta"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    1,
    16
   ]
  }
 ],
 "203:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "purpur_block"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "purpur_block"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "150:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    4,
    7,
    11
   ],
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    4,
    2,
    10
   ],
   "to": [
    6,
    8,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    3,
    2,
    11
   ],
   "to": [
    7,
    8,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    10,
    7,
    11
   ],
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    10,
    2,
    10
   ],
   "to": [
    12,
    8,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    9,
    2,
    11
   ],
   "to": [
    13,
    8,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    5,
    2
   ],
   "to": [
    9,
    5,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "to": [
    9,
    6,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "to": [
    10,
    6,
    4
   ]
  }
 ],
 "69:12": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 90,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "from": [
    7,
    1,
    7
   ],
   "xRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   },
   "to": [
    9,
    11,
    9
   ]
  }
 ],
 "178:10": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "59:4": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_4"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "160:5": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_lime"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_lime"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "60:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "dirt"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "farmland_dry"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "dirt"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "dirt"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "dirt"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "132:12": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    0
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    4
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    4
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    8
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    8
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    12
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    12
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    16
   ]
  }
 ],
 "81:13": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "34:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_sticky"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "yRot": 180,
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "180:7": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "197:8": [
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_dark_oak_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_dark_oak_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_dark_oak_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_dark_oak_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_dark_oak_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "34:9": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_sticky"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 270,
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "xRot": 270,
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "115:2": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_1"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "147:10": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "gold_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "gold_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "160:4": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_yellow"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_yellow"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "203:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "purpur_block"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "purpur_block"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "120:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "endframe_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "end_stone"
    },
    "north": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "east": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "west": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    13,
    16
   ]
  }
 ],
 "157:2": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_activator"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_activator"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_activator"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "55:7": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "195:9": [
  {
   "faces": {
    "up": {
     "uv": [
      8.0,
      16.0,
      11.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      3.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      3.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      1.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_upper"
    },
    "west": {
     "uv": [
      0.0,
      1.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_upper"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    15.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0.0,
      5.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      3.0,
      0.0,
      0.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      11.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    },
    "west": {
     "uv": [
      11.0,
      0.0,
      16.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    }
   },
   "from": [
    0.0,
    14.0,
    11.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0.0,
      9.0,
      3.0,
      4.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      3.0,
      0.0,
      0.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      8.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    },
    "west": {
     "uv": [
      3.0,
      0.0,
      8.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    }
   },
   "from": [
    0.0,
    14.0,
    3.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    16.0,
    8.0
   ]
  }
 ],
 "148:15": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "195:4": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      16.0,
      11.0,
      13.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      14.0,
      13.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      13.0,
      14.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "yRot": 90,
   "to": [
    3.0,
    2.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      15.0
     ],
     "cullface": "west",
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    1.0,
    0.0
   ],
   "name": "Element",
   "yRot": 90,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      13.0,
      11.0,
      8.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      12.0,
      14.0,
      7.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      7.0,
      14.0,
      12.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    7.0
   ],
   "name": "Element",
   "yRot": 90,
   "to": [
    3.0,
    2.0,
    12.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      3.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      3.0,
      14.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      14.0,
      3.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 90,
   "to": [
    3.0,
    2.0,
    3.0
   ]
  }
 ],
 "187:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "107:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "104:5": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      12
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "south": {
     "uv": [
      16,
      0,
      0,
      12
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    0,
    -1,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    16,
    11,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      12
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      12
     ],
     "texture": "pumpkin_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    8,
    -1,
    0
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    11,
    16
   ]
  }
 ],
 "154:11": [
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_inside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    10,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    11,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    11,
    0
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    14,
    11,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    0
   ],
   "yRot": 180,
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    14
   ],
   "yRot": 180,
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    4,
    4,
    4
   ],
   "yRot": 180,
   "to": [
    12,
    10,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    6,
    4,
    0
   ],
   "yRot": 180,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "116:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "south",
     "texture": "enchanting_table_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "enchanting_table_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "enchanting_table_bottom"
    },
    "north": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "north",
     "texture": "enchanting_table_side"
    },
    "east": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "east",
     "texture": "enchanting_table_side"
    },
    "west": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "west",
     "texture": "enchanting_table_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    12,
    16
   ]
  }
 ],
 "193:3": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_spruce_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_spruce_lower"
    },
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_spruce_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_spruce_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_spruce_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "120:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "endframe_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "end_stone"
    },
    "north": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "east": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "west": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    13,
    16
   ]
  }
 ],
 "107:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "117:7": [
  {
   "faces": {
    "south": {
     "uv": [
      7,
      2,
      9,
      16
     ],
     "texture": "brewing_stand"
    },
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "brewing_stand"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "brewing_stand"
    },
    "north": {
     "uv": [
      7,
      2,
      9,
      16
     ],
     "texture": "brewing_stand"
    },
    "east": {
     "uv": [
      7,
      2,
      9,
      16
     ],
     "texture": "brewing_stand"
    },
    "west": {
     "uv": [
      7,
      2,
      9,
      16
     ],
     "texture": "brewing_stand"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    14,
    9
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      9,
      14,
      15,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "up": {
     "uv": [
      9,
      5,
      15,
      11
     ],
     "texture": "brewing_stand_base"
    },
    "down": {
     "uv": [
      9,
      5,
      15,
      11
     ],
     "texture": "brewing_stand_base"
    },
    "north": {
     "uv": [
      9,
      14,
      15,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "east": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "west": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "brewing_stand_base"
    }
   },
   "from": [
    9,
    0,
    5
   ],
   "to": [
    15,
    2,
    11
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      2,
      14,
      8,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "up": {
     "uv": [
      2,
      1,
      8,
      7
     ],
     "texture": "brewing_stand_base"
    },
    "down": {
     "uv": [
      2,
      1,
      8,
      7
     ],
     "texture": "brewing_stand_base"
    },
    "north": {
     "uv": [
      2,
      14,
      8,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "east": {
     "uv": [
      1,
      14,
      7,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "west": {
     "uv": [
      1,
      14,
      7,
      16
     ],
     "texture": "brewing_stand_base"
    }
   },
   "from": [
    2,
    0,
    1
   ],
   "to": [
    8,
    2,
    7
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      2,
      14,
      8,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "up": {
     "uv": [
      2,
      9,
      8,
      15
     ],
     "texture": "brewing_stand_base"
    },
    "down": {
     "uv": [
      2,
      9,
      8,
      15
     ],
     "texture": "brewing_stand_base"
    },
    "north": {
     "uv": [
      2,
      14,
      8,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "east": {
     "uv": [
      9,
      14,
      15,
      16
     ],
     "texture": "brewing_stand_base"
    },
    "west": {
     "uv": [
      9,
      14,
      15,
      16
     ],
     "texture": "brewing_stand_base"
    }
   },
   "from": [
    2,
    0,
    9
   ],
   "to": [
    8,
    2,
    15
   ]
  }
 ],
 "93:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    6
   ],
   "yRot": 180,
   "to": [
    9,
    7,
    8
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "yRot": 180,
   "to": [
    9,
    7,
    4
   ]
  }
 ],
 "178:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "148:8": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "66:4": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_normal"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_normal"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_normal"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_normal"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_normal"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_normal"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "171:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "south",
     "texture": "wool_colored_yellow"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wool_colored_yellow"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "wool_colored_yellow"
    },
    "north": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "north",
     "texture": "wool_colored_yellow"
    },
    "east": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "east",
     "texture": "wool_colored_yellow"
    },
    "west": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "west",
     "texture": "wool_colored_yellow"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    1,
    16
   ]
  }
 ],
 "77:3": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "stone"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "stone"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "stone"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "stone"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "stone"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "stone"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 180,
   "to": [
    11,
    2,
    10
   ]
  }
 ],
 "71:11": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_iron_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "40:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "mushroom_red"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "mushroom_red"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "mushroom_red"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "mushroom_red"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "55:3": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "157:8": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    2,
    0,
    -0.25
   ],
   "to": [
    4,
    1,
    16.25
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    12,
    0,
    -0.25
   ],
   "to": [
    14,
    1,
    16.25
   ]
  }
 ],
 "76:1": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    -1,
    3.5,
    7
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   },
   "to": [
    1,
    13.5,
    9
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    -1,
    3.5,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   },
   "to": [
    1,
    19.5,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    -8,
    3.5,
    7
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     0,
     3.5,
     8
    ],
    "axis": "z"
   },
   "to": [
    8,
    19.5,
    9
   ],
   "shade": false
  }
 ],
 "26:1": [
  {
   "faces": {
    "east": {
     "uv": [
      16,
      7,
      0,
      16
     ],
     "texture": "bed_feet_side"
    },
    "north": {
     "uv": [
      0,
      7,
      16,
      16
     ],
     "texture": "bed_feet_end"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      0
     ],
     "rotation": 90,
     "texture": "bed_feet_top"
    },
    "west": {
     "uv": [
      0,
      7,
      16,
      16
     ],
     "texture": "bed_feet_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    16
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    3,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    3,
    16
   ]
  }
 ],
 "186:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "6:1": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_spruce"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_spruce"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_spruce"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_spruce"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "171:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "south",
     "texture": "wool_colored_silver"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wool_colored_silver"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "wool_colored_silver"
    },
    "north": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "north",
     "texture": "wool_colored_silver"
    },
    "east": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "east",
     "texture": "wool_colored_silver"
    },
    "west": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "west",
     "texture": "wool_colored_silver"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    1,
    16
   ]
  }
 ],
 "109:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stonebrick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "stonebrick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "71:8": [
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_iron_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "107:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "6:2": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_birch"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_birch"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_birch"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_birch"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "127:1": [
  {
   "faces": {
    "south": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "up": {
     "uv": [
      0,
      0,
      4,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "down": {
     "uv": [
      0,
      0,
      4,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "north": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "east": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "west": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    }
   },
   "from": [
    6,
    7,
    11
   ],
   "yRot": 90,
   "to": [
    10,
    12,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      12,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "west": {
     "uv": [
      12,
      0,
      16,
      4
     ],
     "texture": "cocoa_stage_0"
    }
   },
   "from": [
    8,
    12,
    12
   ],
   "yRot": 90,
   "to": [
    8,
    16,
    16
   ]
  }
 ],
 "69:3": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 90,
   "yRot": 180,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 90,
   "yRot": 180,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "195:8": [
  {
   "faces": {
    "up": {
     "uv": [
      8.0,
      16.0,
      11.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      3.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      3.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      1.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_upper"
    },
    "west": {
     "uv": [
      0.0,
      1.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_upper"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    15.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0.0,
      5.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      3.0,
      0.0,
      0.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      11.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    },
    "west": {
     "uv": [
      11.0,
      0.0,
      16.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    }
   },
   "from": [
    0.0,
    14.0,
    11.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0.0,
      9.0,
      3.0,
      4.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      3.0,
      0.0,
      0.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      8.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    },
    "west": {
     "uv": [
      3.0,
      0.0,
      8.0,
      2.0
     ],
     "texture": "door_jungle_upper"
    }
   },
   "from": [
    0.0,
    14.0,
    3.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    16.0,
    8.0
   ]
  }
 ],
 "114:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "nether_brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "nether_brick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "nether_brick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "nether_brick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "nether_brick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "nether_brick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "nether_brick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "nether_brick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "nether_brick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "nether_brick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "nether_brick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "141:3": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "106:11": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    15.2,
    0,
    0
   ],
   "yRot": 90,
   "shade": false,
   "to": [
    15.2,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    15.2
   ],
   "yRot": 90,
   "shade": false,
   "to": [
    16,
    16,
    15.2
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    0.8
   ],
   "yRot": 90,
   "shade": false,
   "to": [
    16,
    16,
    0.8
   ]
  }
 ],
 "198:0": [
  {
   "faces": {
    "south": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      2,
      6,
      6
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      6,
      6,
      2,
      2
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "xRot": 180,
   "to": [
    10,
    1,
    10
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      2
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      4,
      2,
      2,
      0
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    7,
    1,
    7
   ],
   "xRot": 180,
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "148:5": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "175:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_sunflower_bottom"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_sunflower_bottom"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_sunflower_bottom"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_sunflower_bottom"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "163:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_acacia"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_acacia"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "94:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    8
   ],
   "yRot": 180,
   "to": [
    9,
    7,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    7
   ],
   "yRot": 180,
   "to": [
    9,
    8,
    11
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    8
   ],
   "yRot": 180,
   "to": [
    10,
    8,
    10
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    2
   ],
   "yRot": 180,
   "to": [
    9,
    7,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "yRot": 180,
   "to": [
    9,
    8,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "yRot": 180,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "143:4": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "to": [
    11,
    2,
    10
   ]
  }
 ],
 "131:15": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      6,
      16,
      8
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      8,
      16,
      6
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    0.5,
    0
   ],
   "yRot": 90,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     0,
     0
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    8.25,
    0.5,
    6.7
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    3.4,
    6.7
   ],
   "yRot": 90,
   "to": [
    9.8,
    4.2,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    9.1
   ],
   "yRot": 90,
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "yRot": 90,
   "to": [
    8.6,
    4.2,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.4,
    7.9
   ],
   "yRot": 90,
   "to": [
    7.4,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    3.4,
    7.9
   ],
   "yRot": 90,
   "to": [
    8.6,
    4.2,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "yRot": 90,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "yRot": 90,
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "195:2": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      16.0,
      11.0,
      13.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      14.0,
      13.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      13.0,
      14.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      15.0
     ],
     "cullface": "west",
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    1.0,
    0.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      13.0,
      11.0,
      8.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      12.0,
      14.0,
      7.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      7.0,
      14.0,
      12.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    7.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    12.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      3.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      3.0,
      14.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      14.0,
      3.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    3.0
   ]
  }
 ],
 "196:5": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      16.0
     ],
     "cullface": "north",
     "texture": "door_acacia_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      2.0,
      0.0,
      0.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      2.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    16.0,
    2.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_acacia_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      14.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      14.0,
      0.0,
      16.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    14.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      8.0,
      2.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_side"
    },
    "down": {
     "uv": [
      8.0,
      14.0,
      11.0,
      2.0
     ],
     "cullface": "down",
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      14.0,
      14.0,
      2.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      14.0,
      14.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    14.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    3.0,
    3.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    13.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    3.0,
    14.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    5.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    4.0,
    6.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    10.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    4.0,
    11.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      11.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "up": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      4.0,
      12.0,
      5.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    4.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    4.0,
    12.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    5.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    6.0,
    4.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    5.0,
    12.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    6.0,
    14.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    9.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    10.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    6.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    7.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    3.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    4.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      12.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    12.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    13.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      11.0,
      11.0,
      12.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    11.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    5.0,
    13.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      4.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    3.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    5.0,
    5.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    13.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    14.0,
    14.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    9.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    10.0,
    14.0
   ]
  }
 ],
 "34:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "yRot": 180,
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "157:11": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    2,
    7.25,
    0
   ],
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16.75
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    12,
    7.25,
    0
   ],
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16.75
   ]
  }
 ],
 "34:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "134:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_spruce"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_spruce"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_spruce"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_spruce"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_spruce"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_spruce"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_spruce"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_spruce"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_spruce"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_spruce"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_spruce"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "145:9": [
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    2,
    0,
    2
   ],
   "yRot": 90,
   "to": [
    14,
    3,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "anvil_side"
    },
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    3,
    3
   ],
   "yRot": 90,
   "to": [
    13,
    5,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_top_damaged_2"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    13,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    5,
    5,
    4
   ],
   "yRot": 90,
   "to": [
    11,
    10,
    12
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    },
    "down": {
     "texture": "anvil_base"
    }
   },
   "from": [
    4,
    10,
    1
   ],
   "yRot": 90,
   "to": [
    12,
    12,
    15
   ]
  }
 ],
 "163:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_acacia"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_acacia"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "150:10": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    4,
    7,
    11
   ],
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    4,
    2,
    10
   ],
   "to": [
    6,
    8,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    3,
    2,
    11
   ],
   "to": [
    7,
    8,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    10,
    7,
    11
   ],
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    10,
    2,
    10
   ],
   "to": [
    12,
    8,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    9,
    2,
    11
   ],
   "to": [
    13,
    8,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "to": [
    9,
    4,
    4
   ]
  }
 ],
 "160:2": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_magenta"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_magenta"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "194:2": [
  {
   "faces": {
    "north": {
     "uv": [
      11,
      0,
      8,
      16
     ],
     "texture": "door_birch_side"
    },
    "south": {
     "uv": [
      8,
      0,
      11,
      16
     ],
     "texture": "door_birch_side"
    },
    "east": {
     "uv": [
      2,
      0,
      0,
      16
     ],
     "texture": "door_birch_lower"
    },
    "west": {
     "cullface": "west",
     "texture": "door_birch_lower"
    },
    "down": {
     "uv": [
      8,
      14,
      11,
      16
     ],
     "cullface": "down",
     "texture": "door_birch_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    3,
    16,
    2
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      0,
      8,
      16
     ],
     "texture": "door_birch_side"
    },
    "south": {
     "uv": [
      8,
      0,
      11,
      16
     ],
     "texture": "door_birch_side"
    },
    "east": {
     "uv": [
      16,
      0,
      14,
      16
     ],
     "texture": "door_birch_lower"
    },
    "west": {
     "cullface": "west",
     "texture": "door_birch_lower"
    },
    "down": {
     "uv": [
      0,
      0,
      3,
      2
     ],
     "cullface": "down",
     "texture": "door_birch_side"
    }
   },
   "from": [
    0,
    0,
    14
   ],
   "yRot": 180,
   "to": [
    3,
    16,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      14,
      9,
      2,
      16
     ],
     "texture": "door_birch_lower"
    },
    "west": {
     "uv": [
      14,
      9,
      2,
      16
     ],
     "cullface": "west",
     "texture": "door_birch_lower"
    },
    "up": {
     "texture": "door_birch_side"
    },
    "down": {
     "cullface": "down",
     "texture": "door_birch_side"
    }
   },
   "from": [
    0,
    0,
    2
   ],
   "yRot": 180,
   "to": [
    3,
    7,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "door_birch_lower"
    },
    "west": {
     "texture": "door_birch_lower"
    }
   },
   "from": [
    1,
    0,
    2
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    14
   ]
  }
 ],
 "64:5": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_wood_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "77:10": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "stone"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "stone"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "stone"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "stone"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "stone"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      15
     ],
     "texture": "stone"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 270,
   "to": [
    11,
    1,
    10
   ]
  }
 ],
 "126:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "78:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "snow"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "snow"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "snow"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "snow"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "snow"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "snow"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "127:2": [
  {
   "faces": {
    "south": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "up": {
     "uv": [
      0,
      0,
      4,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "down": {
     "uv": [
      0,
      0,
      4,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "north": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "east": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "west": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    }
   },
   "from": [
    6,
    7,
    11
   ],
   "yRot": 180,
   "to": [
    10,
    12,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      12,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "west": {
     "uv": [
      12,
      0,
      16,
      4
     ],
     "texture": "cocoa_stage_0"
    }
   },
   "from": [
    8,
    12,
    12
   ],
   "yRot": 180,
   "to": [
    8,
    16,
    16
   ]
  }
 ],
 "132:5": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    0
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    4
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    4
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    8
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    8
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    12
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      2,
      16,
      4
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      2
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    12
   ],
   "shade": false,
   "to": [
    8.25,
    1.5,
    16
   ]
  }
 ],
 "69:0": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 180,
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "127:5": [
  {
   "faces": {
    "south": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    },
    "up": {
     "uv": [
      0,
      0,
      6,
      6
     ],
     "texture": "cocoa_stage_1"
    },
    "down": {
     "uv": [
      0,
      0,
      6,
      6
     ],
     "texture": "cocoa_stage_1"
    },
    "north": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    },
    "east": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    },
    "west": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    }
   },
   "from": [
    5,
    5,
    9
   ],
   "yRot": 90,
   "to": [
    11,
    12,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      12,
      4
     ],
     "texture": "cocoa_stage_1"
    },
    "west": {
     "uv": [
      12,
      0,
      16,
      4
     ],
     "texture": "cocoa_stage_1"
    }
   },
   "from": [
    8,
    12,
    12
   ],
   "yRot": 90,
   "to": [
    8,
    16,
    16
   ]
  }
 ],
 "131:10": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    4.2,
    6.7
   ],
   "to": [
    9.8,
    5,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    9.1
   ],
   "to": [
    8.6,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    7.9
   ],
   "to": [
    8.6,
    5,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    7.9
   ],
   "to": [
    7.4,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    4.2,
    7.9
   ],
   "to": [
    8.6,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "105:1": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    },
    "south": {
     "uv": [
      16,
      0,
      0,
      4
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    0,
    -1,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    16,
    3,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      4
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    8,
    -1,
    0
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    3,
    16
   ]
  }
 ],
 "6:5": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "185:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "188:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_spruce"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "up",
     "texture": "fence_spruce"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "fence_spruce"
    },
    "north": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_spruce"
    },
    "east": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_spruce"
    },
    "west": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_spruce"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "__comment": "Center post",
   "to": [
    10,
    16,
    10
   ]
  }
 ],
 "135:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_birch"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_birch"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "28:8": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_detector_powered"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    }
   },
   "from": [
    2,
    0,
    -0.25
   ],
   "to": [
    4,
    1,
    16.25
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector_powered"
    }
   },
   "from": [
    12,
    0,
    -0.25
   ],
   "to": [
    14,
    1,
    16.25
   ]
  }
 ],
 "109:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stonebrick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "stonebrick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "64:7": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_wood_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "26:11": [
  {
   "faces": {
    "east": {
     "uv": [
      15,
      7,
      0,
      16
     ],
     "texture": "bed_head_side"
    },
    "south": {
     "uv": [
      0,
      7,
      16,
      16
     ],
     "texture": "bed_head_end_tall"
    },
    "up": {
     "uv": [
      0,
      0,
      15,
      16
     ],
     "rotation": 90,
     "texture": "bed_head_top"
    },
    "west": {
     "uv": [
      0,
      7,
      15,
      16
     ],
     "texture": "bed_head_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    9,
    15
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    3,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    3,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "bed_head_end_tall"
    },
    "north": {
     "texture": "bed_head_end_tall"
    },
    "south": {
     "texture": "bed_head_end_tall"
    },
    "up": {
     "uv": [
      0,
      3,
      16,
      4
     ],
     "texture": "bed_head_end_tall"
    },
    "west": {
     "texture": "bed_head_end_tall"
    }
   },
   "from": [
    0,
    0,
    15
   ],
   "yRot": 270,
   "to": [
    16,
    13,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      14,
      3,
      15,
      4
     ],
     "texture": "bed_head_end_tall"
    },
    "north": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "bed_head_end_tall"
    },
    "south": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "bed_head_end_tall"
    },
    "up": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "bed_head_end_tall"
    },
    "west": {
     "uv": [
      1,
      3,
      2,
      4
     ],
     "texture": "bed_head_end_tall"
    }
   },
   "from": [
    1,
    13,
    15
   ],
   "yRot": 270,
   "to": [
    15,
    14,
    16
   ]
  }
 ],
 "147:2": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "gold_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "gold_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "171:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "south",
     "texture": "wool_colored_pink"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wool_colored_pink"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "wool_colored_pink"
    },
    "north": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "north",
     "texture": "wool_colored_pink"
    },
    "east": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "east",
     "texture": "wool_colored_pink"
    },
    "west": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "west",
     "texture": "wool_colored_pink"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    1,
    16
   ]
  }
 ],
 "184:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "78:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "snow"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "snow"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "snow"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "snow"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "snow"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "snow"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  }
 ],
 "167:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "iron_trapdoor"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "up",
     "texture": "iron_trapdoor"
    },
    "down": {
     "uv": [
      0,
      13,
      16,
      16
     ],
     "cullface": "down",
     "texture": "iron_trapdoor"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "iron_trapdoor"
    },
    "east": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "iron_trapdoor"
    },
    "west": {
     "uv": [
      16,
      0,
      13,
      16
     ],
     "cullface": "west",
     "texture": "iron_trapdoor"
    }
   },
   "from": [
    0,
    0,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "115:3": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_wart_stage_2"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "55:9": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "160:0": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_white"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_white"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "55:6": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "96:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "trapdoor"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "up",
     "texture": "trapdoor"
    },
    "down": {
     "uv": [
      0,
      13,
      16,
      16
     ],
     "cullface": "down",
     "texture": "trapdoor"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "trapdoor"
    },
    "east": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "trapdoor"
    },
    "west": {
     "uv": [
      16,
      0,
      13,
      16
     ],
     "cullface": "west",
     "texture": "trapdoor"
    }
   },
   "from": [
    0,
    0,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "151:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "106:3": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    0.8
   ],
   "yRot": 180,
   "shade": false,
   "to": [
    16,
    16,
    0.8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    15.2,
    0,
    0
   ],
   "yRot": 180,
   "shade": false,
   "to": [
    15.2,
    16,
    16
   ]
  }
 ],
 "60:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "dirt"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "farmland_dry"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "dirt"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "dirt"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "dirt"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "6:13": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sapling_roofed_oak"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "53:7": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "184:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "83:9": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "126:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_birch"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "197:11": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_dark_oak_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_dark_oak_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_dark_oak_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_dark_oak_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_dark_oak_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "34:12": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_top_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "piston_top_sticky"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side",
     "cullface": "east"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side",
     "cullface": "west"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "texture": "piston_side"
    },
    "west": {
     "uv": [
      16,
      4,
      0,
      0
     ],
     "texture": "piston_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 270,
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      4
     ],
     "rotation": 90,
     "texture": "piston_side"
    }
   },
   "from": [
    6,
    6,
    4
   ],
   "yRot": 270,
   "to": [
    10,
    10,
    20
   ]
  }
 ],
 "180:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "red_sandstone_normal"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "red_sandstone_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "red_sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "red_sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "red_sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "red_sandstone_normal"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "2:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "grass_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "grass_top",
     "tintindex": 0
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "grass_side"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "grass_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "grass_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "north",
     "texture": "grass_side_overlay",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "grass_side_overlay",
     "tintindex": 0
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "grass_side_overlay",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "grass_side_overlay",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "145:3": [
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    2,
    0,
    2
   ],
   "yRot": 270,
   "to": [
    14,
    3,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "anvil_side"
    },
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    3,
    3
   ],
   "yRot": 270,
   "to": [
    13,
    5,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_top_damaged_0"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    13,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    5,
    5,
    4
   ],
   "yRot": 270,
   "to": [
    11,
    10,
    12
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    },
    "down": {
     "texture": "anvil_base"
    }
   },
   "from": [
    4,
    10,
    1
   ],
   "yRot": 270,
   "to": [
    12,
    12,
    15
   ]
  }
 ],
 "147:9": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "gold_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "gold_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "184:15": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "187:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "122:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      15,
      10,
      16
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      6,
      15,
      10,
      16
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      6,
      15,
      10,
      16
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      6,
      15,
      10,
      16
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    6,
    15,
    6
   ],
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      5,
      5,
      11,
      11
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      5,
      5,
      11,
      11
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      5,
      14,
      11,
      15
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    5,
    14,
    5
   ],
   "to": [
    11,
    15,
    11
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      4,
      13,
      12,
      14
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      4,
      4,
      12,
      12
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      4,
      4,
      12,
      12
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      4,
      13,
      12,
      14
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      4,
      13,
      12,
      14
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      4,
      13,
      12,
      14
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    5,
    13,
    5
   ],
   "to": [
    11,
    14,
    11
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      3,
      11,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      3,
      3,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      3,
      3,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      3,
      11,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      3,
      11,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      3,
      11,
      13,
      13
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    3,
    11,
    3
   ],
   "to": [
    13,
    13,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      2,
      8,
      14,
      11
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      2,
      8,
      14,
      11
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      2,
      8,
      14,
      11
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      2,
      8,
      14,
      11
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    2,
    8,
    2
   ],
   "to": [
    14,
    11,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      1,
      3,
      15,
      8
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      1,
      3,
      15,
      8
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      1,
      3,
      15,
      8
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      1,
      3,
      15,
      8
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    1,
    3,
    1
   ],
   "to": [
    15,
    8,
    15
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      2,
      1,
      14,
      3
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      2,
      1,
      14,
      3
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      2,
      1,
      14,
      3
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      2,
      1,
      14,
      3
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    2,
    1,
    2
   ],
   "to": [
    14,
    3,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      3,
      0,
      13,
      1
     ],
     "texture": "dragon_egg"
    },
    "up": {
     "uv": [
      3,
      3,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "down": {
     "uv": [
      3,
      3,
      13,
      13
     ],
     "texture": "dragon_egg"
    },
    "north": {
     "uv": [
      3,
      0,
      13,
      1
     ],
     "texture": "dragon_egg"
    },
    "east": {
     "uv": [
      3,
      0,
      13,
      1
     ],
     "texture": "dragon_egg"
    },
    "west": {
     "uv": [
      3,
      0,
      13,
      1
     ],
     "texture": "dragon_egg"
    }
   },
   "from": [
    3,
    0,
    3
   ],
   "to": [
    13,
    1,
    13
   ]
  }
 ],
 "83:6": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "187:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "59:7": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wheat_stage_7"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "51:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "194:10": [
  {
   "faces": {
    "east": {
     "uv": [
      2,
      0,
      0,
      16
     ],
     "texture": "door_birch_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "texture": "door_birch_side"
    },
    "south": {
     "texture": "door_birch_side"
    },
    "up": {
     "cullface": "up",
     "texture": "door_birch_side"
    },
    "west": {
     "cullface": "west",
     "texture": "door_birch_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    2
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      14,
      16
     ],
     "texture": "door_birch_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "texture": "door_birch_side"
    },
    "south": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "texture": "door_birch_side"
    },
    "up": {
     "uv": [
      8,
      14,
      11,
      16
     ],
     "cullface": "up",
     "texture": "door_birch_side"
    },
    "west": {
     "cullface": "west",
     "texture": "door_birch_upper"
    }
   },
   "from": [
    0,
    0,
    14
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      14,
      0,
      2,
      2
     ],
     "texture": "door_birch_upper"
    },
    "west": {
     "uv": [
      14,
      0,
      2,
      2
     ],
     "cullface": "west",
     "texture": "door_birch_upper"
    },
    "up": {
     "cullface": "up",
     "texture": "door_birch_side"
    },
    "down": {
     "texture": "door_birch_side"
    }
   },
   "from": [
    0,
    14,
    2
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "door_birch_upper"
    },
    "west": {
     "texture": "door_birch_upper"
    }
   },
   "from": [
    1,
    0,
    2
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    14
   ]
  }
 ],
 "78:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      12,
      16,
      16
     ],
     "cullface": "south",
     "texture": "snow"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "snow"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "snow"
    },
    "north": {
     "uv": [
      0,
      12,
      16,
      16
     ],
     "cullface": "north",
     "texture": "snow"
    },
    "east": {
     "uv": [
      0,
      12,
      16,
      16
     ],
     "cullface": "east",
     "texture": "snow"
    },
    "west": {
     "uv": [
      0,
      12,
      16,
      16
     ],
     "cullface": "west",
     "texture": "snow"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    4,
    16
   ]
  }
 ],
 "184:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_birch"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_birch"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "69:15": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 180,
   "yRot": 180,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "96:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "trapdoor"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "up",
     "texture": "trapdoor"
    },
    "down": {
     "uv": [
      0,
      13,
      16,
      16
     ],
     "cullface": "down",
     "texture": "trapdoor"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "trapdoor"
    },
    "east": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "trapdoor"
    },
    "west": {
     "uv": [
      16,
      0,
      13,
      16
     ],
     "cullface": "west",
     "texture": "trapdoor"
    }
   },
   "from": [
    0,
    0,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "65:4": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "ladder"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "ladder"
    }
   },
   "from": [
    0,
    0,
    15.2
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    15.2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "ladder"
    },
    "up": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "ladder"
    },
    "down": {
     "uv": [
      1,
      3,
      15,
      4
     ],
     "texture": "ladder"
    },
    "north": {
     "texture": "ladder"
    },
    "east": {
     "uv": [
      14,
      2,
      15,
      4
     ],
     "texture": "ladder"
    },
    "west": {
     "uv": [
      14,
      2,
      15,
      4
     ],
     "texture": "ladder"
    }
   },
   "from": [
    1,
    12,
    14.5
   ],
   "yRot": 270,
   "to": [
    15,
    14,
    15.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "ladder"
    },
    "up": {
     "uv": [
      1,
      10,
      15,
      11
     ],
     "texture": "ladder"
    },
    "down": {
     "uv": [
      1,
      11,
      15,
      12
     ],
     "texture": "ladder"
    },
    "north": {
     "texture": "ladder"
    },
    "east": {
     "uv": [
      14,
      10,
      15,
      12
     ],
     "texture": "ladder"
    },
    "west": {
     "uv": [
      14,
      10,
      15,
      12
     ],
     "texture": "ladder"
    }
   },
   "from": [
    1,
    4,
    14.5
   ],
   "yRot": 270,
   "to": [
    15,
    6,
    15.5
   ]
  }
 ],
 "183:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 90,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 90,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "157:12": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "33:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "piston_bottom"
    },
    "up": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_inner"
    },
    "east": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "east",
     "texture": "piston_side",
     "rotation": 90
    },
    "west": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "west",
     "texture": "piston_side",
     "rotation": 270
    }
   },
   "from": [
    0,
    0,
    4
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "66:2": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_normal"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_normal"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_normal"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_normal"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_normal"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_normal"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_normal"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "yRot": 90,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "38:1": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_blue_orchid"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_blue_orchid"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_blue_orchid"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_blue_orchid"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "106:13": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    15.2,
    0,
    0
   ],
   "shade": false,
   "to": [
    15.2,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    15.2
   ],
   "shade": false,
   "to": [
    16,
    16,
    15.2
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    0.8
   ],
   "shade": false,
   "to": [
    16,
    16,
    0.8
   ]
  }
 ],
 "187:10": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 180,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "175:2": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_grass_bottom",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_grass_bottom",
     "tintindex": 0
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_grass_bottom",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "double_plant_grass_bottom",
     "tintindex": 0
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "143:2": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 270,
   "to": [
    11,
    2,
    10
   ]
  }
 ],
 "93:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    10
   ],
   "to": [
    9,
    7,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "to": [
    9,
    7,
    4
   ]
  }
 ],
 "83:3": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "157:13": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    2,
    7.25,
    0
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16.75
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    12,
    7.25,
    0
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16.75
   ]
  }
 ],
 "187:15": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 270,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "198:3": [
  {
   "faces": {
    "south": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      2,
      6,
      6
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      6,
      6,
      2,
      2
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 180,
   "to": [
    10,
    1,
    10
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      2
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      4,
      2,
      2,
      0
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    7,
    1,
    7
   ],
   "xRot": 90,
   "yRot": 180,
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "136:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_jungle"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_jungle"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "187:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "55:12": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "203:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "purpur_block"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "purpur_block"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "127:6": [
  {
   "faces": {
    "south": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    },
    "up": {
     "uv": [
      0,
      0,
      6,
      6
     ],
     "texture": "cocoa_stage_1"
    },
    "down": {
     "uv": [
      0,
      0,
      6,
      6
     ],
     "texture": "cocoa_stage_1"
    },
    "north": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    },
    "east": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    },
    "west": {
     "uv": [
      9,
      4,
      15,
      11
     ],
     "texture": "cocoa_stage_1"
    }
   },
   "from": [
    5,
    5,
    9
   ],
   "yRot": 180,
   "to": [
    11,
    12,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      12,
      4
     ],
     "texture": "cocoa_stage_1"
    },
    "west": {
     "uv": [
      12,
      0,
      16,
      4
     ],
     "texture": "cocoa_stage_1"
    }
   },
   "from": [
    8,
    12,
    12
   ],
   "yRot": 180,
   "to": [
    8,
    16,
    16
   ]
  }
 ],
 "69:9": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 90,
   "yRot": 90,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 90,
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "156:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "quartz_block_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "quartz_block_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "quartz_block_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "quartz_block_side"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "quartz_block_side"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "quartz_block_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "quartz_block_side"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "quartz_block_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "quartz_block_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "quartz_block_side"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "quartz_block_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "quartz_block_side"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "142:2": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "potatoes_stage_1"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "27:11": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    2,
    7.25,
    0
   ],
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16.75
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    12,
    7.25,
    0
   ],
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16.75
   ]
  }
 ],
 "151:10": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "136:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_jungle"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_jungle"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "64:6": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_wood_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "186:12": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "136:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_jungle"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_jungle"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "53:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "51:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "38:4": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_red"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_red"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_red"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_tulip_red"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "189:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_birch"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "up",
     "texture": "fence_birch"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "fence_birch"
    },
    "north": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_birch"
    },
    "east": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_birch"
    },
    "west": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_birch"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "__comment": "Center post",
   "to": [
    10,
    16,
    10
   ]
  }
 ],
 "197:3": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_dark_oak_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_dark_oak_lower"
    },
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_dark_oak_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_dark_oak_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_dark_oak_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "44:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "27:4": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_golden"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_golden"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_golden"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "195:3": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      16.0,
      11.0,
      13.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      14.0,
      13.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      13.0,
      14.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    2.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      15.0
     ],
     "cullface": "west",
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    1.0,
    0.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      13.0,
      11.0,
      8.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      12.0,
      14.0,
      7.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      7.0,
      14.0,
      12.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    7.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    2.0,
    12.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      3.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      3.0,
      14.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      14.0,
      3.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    2.0,
    3.0
   ]
  }
 ],
 "128:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "sandstone_normal"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "163:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_acacia"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_acacia"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "200:2": [
  {
   "faces": {
    "east": {
     "uv": [
      2,
      0,
      14,
      2
     ],
     "texture": "chorus_plant"
    },
    "north": {
     "uv": [
      2,
      0,
      14,
      2
     ],
     "texture": "chorus_plant"
    },
    "south": {
     "uv": [
      2,
      0,
      14,
      2
     ],
     "texture": "chorus_plant"
    },
    "up": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "chorus_flower"
    },
    "west": {
     "uv": [
      2,
      0,
      14,
      2
     ],
     "texture": "chorus_plant"
    }
   },
   "from": [
    2,
    14,
    2
   ],
   "to": [
    14,
    16,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      14,
      2,
      16,
      14
     ],
     "texture": "chorus_plant"
    },
    "south": {
     "uv": [
      0,
      2,
      2,
      14
     ],
     "texture": "chorus_plant"
    },
    "up": {
     "uv": [
      0,
      2,
      2,
      14
     ],
     "texture": "chorus_plant"
    },
    "west": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "chorus_flower"
    },
    "down": {
     "uv": [
      16,
      14,
      14,
      2
     ],
     "texture": "chorus_plant"
    }
   },
   "from": [
    0,
    2,
    2
   ],
   "to": [
    2,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      14,
      2,
      16,
      14
     ],
     "texture": "chorus_plant"
    },
    "north": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "chorus_flower"
    },
    "west": {
     "uv": [
      0,
      2,
      2,
      14
     ],
     "texture": "chorus_plant"
    },
    "up": {
     "uv": [
      2,
      0,
      14,
      2
     ],
     "texture": "chorus_plant"
    },
    "down": {
     "uv": [
      14,
      2,
      2,
      0
     ],
     "texture": "chorus_plant"
    }
   },
   "from": [
    2,
    2,
    0
   ],
   "to": [
    14,
    14,
    2
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      2,
      2,
      14
     ],
     "texture": "chorus_plant"
    },
    "south": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "chorus_flower"
    },
    "up": {
     "uv": [
      2,
      14,
      14,
      16
     ],
     "texture": "chorus_plant"
    },
    "west": {
     "uv": [
      14,
      2,
      16,
      14
     ],
     "texture": "chorus_plant"
    },
    "down": {
     "uv": [
      14,
      16,
      2,
      14
     ],
     "texture": "chorus_plant"
    }
   },
   "from": [
    2,
    2,
    14
   ],
   "to": [
    14,
    14,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "chorus_flower"
    },
    "north": {
     "uv": [
      0,
      2,
      2,
      14
     ],
     "texture": "chorus_plant"
    },
    "south": {
     "uv": [
      14,
      2,
      16,
      14
     ],
     "texture": "chorus_plant"
    },
    "up": {
     "uv": [
      14,
      2,
      16,
      14
     ],
     "texture": "chorus_plant"
    },
    "down": {
     "uv": [
      2,
      14,
      0,
      2
     ],
     "texture": "chorus_plant"
    }
   },
   "from": [
    14,
    2,
    2
   ],
   "to": [
    16,
    14,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      2,
      2,
      14,
      16
     ],
     "texture": "chorus_plant"
    },
    "up": {
     "uv": [
      2,
      2,
      14,
      14
     ],
     "texture": "chorus_plant"
    },
    "down": {
     "uv": [
      14,
      14,
      2,
      2
     ],
     "texture": "chorus_plant"
    },
    "north": {
     "uv": [
      2,
      2,
      14,
      16
     ],
     "texture": "chorus_plant"
    },
    "east": {
     "uv": [
      2,
      2,
      14,
      16
     ],
     "texture": "chorus_plant"
    },
    "west": {
     "uv": [
      2,
      2,
      14,
      16
     ],
     "texture": "chorus_plant"
    }
   },
   "from": [
    2,
    0,
    2
   ],
   "to": [
    14,
    14,
    14
   ]
  }
 ],
 "44:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "west",
     "texture": "stonebrick"
    }
   },
   "from": [
    0,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "53:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "136:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_jungle"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_jungle"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "53:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "195:5": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      16.0,
      11.0,
      13.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      14.0,
      13.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      13.0,
      14.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      15.0
     ],
     "cullface": "west",
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    1.0,
    0.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      13.0,
      11.0,
      8.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      12.0,
      14.0,
      7.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      7.0,
      14.0,
      12.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    7.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    12.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      3.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      3.0,
      14.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      14.0,
      3.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    3.0
   ]
  }
 ],
 "135:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_birch"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_birch"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "92:2": [
  {
   "faces": {
    "south": {
     "texture": "cake_side"
    },
    "up": {
     "texture": "cake_top"
    },
    "down": {
     "cullface": "down",
     "texture": "cake_bottom"
    },
    "north": {
     "texture": "cake_side"
    },
    "east": {
     "texture": "cake_side"
    },
    "west": {
     "texture": "cake_inner"
    }
   },
   "from": [
    5,
    0,
    1
   ],
   "to": [
    15,
    8,
    15
   ]
  }
 ],
 "131:4": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      6,
      16,
      8
     ],
     "rotation": 90,
     "texture": "trip_wire"
    },
    "down": {
     "uv": [
      0,
      8,
      16,
      6
     ],
     "rotation": 90,
     "texture": "trip_wire"
    }
   },
   "from": [
    7.75,
    1.5,
    0
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     0,
     0
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    8.25,
    1.5,
    6.7
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    4.2,
    6.7
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     4.2,
     6.7
    ],
    "axis": "x",
    "rescale": false
   },
   "to": [
    9.8,
    5,
    10.3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    9.1
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     4.2,
     6.7
    ],
    "axis": "x",
    "rescale": false
   },
   "to": [
    8.6,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    7.9
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     4.2,
     6.7
    ],
    "axis": "x",
    "rescale": false
   },
   "to": [
    8.6,
    5,
    7.9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    4.2,
    7.9
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     4.2,
     6.7
    ],
    "axis": "x",
    "rescale": false
   },
   "to": [
    7.4,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    4.2,
    7.9
   ],
   "yRot": 180,
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     4.2,
     6.7
    ],
    "axis": "x",
    "rescale": false
   },
   "to": [
    8.6,
    5,
    9.1
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "yRot": 180,
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "yRot": 180,
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "154:5": [
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_inside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    10,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    11,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    14,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    14
   ],
   "yRot": 90,
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    4,
    4,
    4
   ],
   "yRot": 90,
   "to": [
    12,
    10,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    6,
    4,
    0
   ],
   "yRot": 90,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "186:9": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 90,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 90,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 90,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "120:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "endframe_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "end_stone"
    },
    "north": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "east": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "west": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    13,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "endframe_eye"
    },
    "up": {
     "uv": [
      4,
      4,
      12,
      12
     ],
     "texture": "endframe_eye"
    },
    "down": {
     "uv": [
      4,
      4,
      12,
      12
     ],
     "texture": "endframe_eye"
    },
    "north": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "endframe_eye"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "endframe_eye"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "endframe_eye"
    }
   },
   "from": [
    4,
    13,
    4
   ],
   "to": [
    12,
    16,
    12
   ]
  }
 ],
 "141:1": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_0"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "113:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_nether_brick"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "up",
     "texture": "fence_nether_brick"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "fence_nether_brick"
    },
    "north": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_nether_brick"
    },
    "east": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_nether_brick"
    },
    "west": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_nether_brick"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "__comment": "Center post",
   "to": [
    10,
    16,
    10
   ]
  }
 ],
 "194:11": [
  {
   "faces": {
    "east": {
     "uv": [
      2,
      0,
      0,
      16
     ],
     "texture": "door_birch_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "texture": "door_birch_side"
    },
    "south": {
     "texture": "door_birch_side"
    },
    "up": {
     "cullface": "up",
     "texture": "door_birch_side"
    },
    "west": {
     "cullface": "west",
     "texture": "door_birch_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    2
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      14,
      16
     ],
     "texture": "door_birch_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "texture": "door_birch_side"
    },
    "south": {
     "texture": "door_birch_side"
    },
    "up": {
     "uv": [
      8,
      14,
      11,
      16
     ],
     "cullface": "up",
     "texture": "door_birch_side"
    },
    "west": {
     "cullface": "west",
     "texture": "door_birch_upper"
    }
   },
   "from": [
    0,
    0,
    14
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      14,
      0,
      2,
      2
     ],
     "texture": "door_birch_upper"
    },
    "west": {
     "uv": [
      14,
      0,
      2,
      2
     ],
     "cullface": "west",
     "texture": "door_birch_upper"
    },
    "up": {
     "cullface": "up",
     "texture": "door_birch_side"
    },
    "down": {
     "texture": "door_birch_side"
    }
   },
   "from": [
    0,
    14,
    2
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "door_birch_upper"
    },
    "west": {
     "texture": "door_birch_upper"
    }
   },
   "from": [
    1,
    0,
    2
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    14
   ]
  }
 ],
 "118:1": [
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    3,
    0
   ],
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_inner"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    2,
    3,
    2
   ],
   "to": [
    14,
    4,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    14,
    3,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    2,
    3,
    0
   ],
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    2,
    3,
    14
   ],
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    4,
    3,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    2
   ],
   "to": [
    2,
    3,
    4
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    12,
    0,
    0
   ],
   "to": [
    16,
    3,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    14,
    0,
    2
   ],
   "to": [
    16,
    3,
    4
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    14
   ],
   "to": [
    4,
    3,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    12
   ],
   "to": [
    2,
    3,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    12,
    0,
    14
   ],
   "to": [
    16,
    3,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    14,
    0,
    12
   ],
   "to": [
    16,
    3,
    14
   ]
  },
  {
   "faces": {
    "up": {
     "texture": "water_still"
    }
   },
   "from": [
    2,
    9,
    2
   ],
   "to": [
    14,
    9,
    14
   ]
  }
 ],
 "60:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "south",
     "texture": "dirt"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "farmland_dry"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "dirt"
    },
    "north": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "north",
     "texture": "dirt"
    },
    "east": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "east",
     "texture": "dirt"
    },
    "west": {
     "uv": [
      0,
      1,
      16,
      16
     ],
     "cullface": "west",
     "texture": "dirt"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    15,
    16
   ]
  }
 ],
 "44:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stone_slab_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_side"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_side"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "185:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "108:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "brick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "brick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "brick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "brick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "brick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "brick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "brick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "38:8": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_oxeye_daisy"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_oxeye_daisy"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_oxeye_daisy"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_oxeye_daisy"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "106:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    15.2
   ],
   "shade": false,
   "to": [
    16,
    16,
    15.2
   ]
  }
 ],
 "128:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "sandstone_normal"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "sandstone_normal"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "sandstone_top"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "sandstone_bottom"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "sandstone_normal"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "sandstone_normal"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "sandstone_normal"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "38:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_rose"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_rose"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_rose"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_rose"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "191:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_big_oak"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "up",
     "texture": "fence_big_oak"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "fence_big_oak"
    },
    "north": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_big_oak"
    },
    "east": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_big_oak"
    },
    "west": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_big_oak"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "__comment": "Center post",
   "to": [
    10,
    16,
    10
   ]
  }
 ],
 "207:2": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "beetroots_stage_2"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "185:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_jungle"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_jungle"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_jungle"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "183:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "77:2": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "stone"
    },
    "up": {
     "uv": [
      5,
      10,
      11,
      6
     ],
     "texture": "stone"
    },
    "down": {
     "uv": [
      5,
      6,
      11,
      10
     ],
     "cullface": "down",
     "texture": "stone"
    },
    "north": {
     "uv": [
      5,
      14,
      11,
      16
     ],
     "texture": "stone"
    },
    "east": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "stone"
    },
    "west": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "stone"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 270,
   "to": [
    11,
    2,
    10
   ]
  }
 ],
 "205:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "purpur_block"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "31:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "deadbush"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "deadbush"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "deadbush"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "deadbush"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "44:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "cobblestone"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "186:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_big_oak"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_big_oak"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "148:10": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "193:11": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_spruce_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_spruce_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_spruce_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_spruce_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_spruce_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "203:7": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "purpur_block"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "purpur_block"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "purpur_block"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "purpur_block"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "purpur_block"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "purpur_block"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "purpur_block"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "141:7": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_3"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "135:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_birch"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_birch"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_birch"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_birch"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_birch"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_birch"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_birch"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "197:9": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_dark_oak_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_dark_oak_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_dark_oak_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_dark_oak_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_dark_oak_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "160:7": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_gray"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_gray"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "44:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "nether_brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "nether_brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_brick"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "north",
     "texture": "nether_brick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "nether_brick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "west",
     "texture": "nether_brick"
    }
   },
   "from": [
    0,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "192:0": [
  {
   "faces": {
    "south": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_acacia"
    },
    "up": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "up",
     "texture": "fence_acacia"
    },
    "down": {
     "uv": [
      6,
      6,
      10,
      10
     ],
     "cullface": "down",
     "texture": "fence_acacia"
    },
    "north": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_acacia"
    },
    "east": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_acacia"
    },
    "west": {
     "uv": [
      6,
      0,
      10,
      16
     ],
     "texture": "fence_acacia"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "__comment": "Center post",
   "to": [
    10,
    16,
    10
   ]
  }
 ],
 "71:5": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_iron_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "92:5": [
  {
   "faces": {
    "south": {
     "texture": "cake_side"
    },
    "up": {
     "texture": "cake_top"
    },
    "down": {
     "cullface": "down",
     "texture": "cake_bottom"
    },
    "north": {
     "texture": "cake_side"
    },
    "east": {
     "texture": "cake_side"
    },
    "west": {
     "texture": "cake_inner"
    }
   },
   "from": [
    11,
    0,
    1
   ],
   "to": [
    15,
    8,
    15
   ]
  }
 ],
 "67:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "cobblestone"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "27:12": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    0,
    8.25,
    1
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    8.25,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    2,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    4,
    8.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_golden_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_golden_powered"
    }
   },
   "from": [
    12,
    7.25,
    -0.75
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     7.25,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    14,
    8.25,
    16
   ]
  }
 ],
 "64:4": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_wood_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "127:0": [
  {
   "faces": {
    "south": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "up": {
     "uv": [
      0,
      0,
      4,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "down": {
     "uv": [
      0,
      0,
      4,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "north": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "east": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    },
    "west": {
     "uv": [
      11,
      4,
      15,
      9
     ],
     "texture": "cocoa_stage_0"
    }
   },
   "from": [
    6,
    7,
    11
   ],
   "to": [
    10,
    12,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      12,
      4
     ],
     "texture": "cocoa_stage_0"
    },
    "west": {
     "uv": [
      12,
      0,
      16,
      4
     ],
     "texture": "cocoa_stage_0"
    }
   },
   "from": [
    8,
    12,
    12
   ],
   "to": [
    8,
    16,
    16
   ]
  }
 ],
 "96:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "trapdoor"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "up",
     "texture": "trapdoor"
    },
    "down": {
     "uv": [
      0,
      13,
      16,
      16
     ],
     "cullface": "down",
     "texture": "trapdoor"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "trapdoor"
    },
    "east": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "trapdoor"
    },
    "west": {
     "uv": [
      16,
      0,
      13,
      16
     ],
     "cullface": "west",
     "texture": "trapdoor"
    }
   },
   "from": [
    0,
    0,
    13
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "29:13": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "piston_bottom"
    },
    "up": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_inner"
    },
    "east": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "east",
     "texture": "piston_side",
     "rotation": 90
    },
    "west": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "west",
     "texture": "piston_side",
     "rotation": 270
    }
   },
   "from": [
    0,
    0,
    4
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "94:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    8
   ],
   "to": [
    9,
    7,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    7
   ],
   "to": [
    9,
    8,
    11
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    8
   ],
   "to": [
    10,
    8,
    10
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    2
   ],
   "to": [
    9,
    7,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "to": [
    9,
    8,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "178:9": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "145:8": [
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    2,
    0,
    2
   ],
   "to": [
    14,
    3,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "anvil_side"
    },
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_base"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    3,
    3
   ],
   "to": [
    13,
    5,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "anvil_side"
    },
    "up": {
     "texture": "anvil_top_damaged_2"
    },
    "down": {
     "texture": "anvil_base"
    },
    "north": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    3,
    11,
    0
   ],
   "to": [
    13,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    }
   },
   "from": [
    5,
    5,
    4
   ],
   "to": [
    11,
    10,
    12
   ]
  },
  {
   "faces": {
    "north": {
     "texture": "anvil_side"
    },
    "south": {
     "texture": "anvil_side"
    },
    "east": {
     "texture": "anvil_side"
    },
    "west": {
     "texture": "anvil_side"
    },
    "down": {
     "texture": "anvil_base"
    }
   },
   "from": [
    4,
    10,
    1
   ],
   "to": [
    12,
    12,
    15
   ]
  }
 ],
 "178:15": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "196:2": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      16.0
     ],
     "cullface": "north",
     "texture": "door_acacia_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      2.0,
      0.0,
      0.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      2.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    16.0,
    2.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      0.0,
      3.0,
      2.0
     ],
     "texture": "door_acacia_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      14.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      14.0,
      0.0,
      16.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    14.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      8.0,
      2.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_side"
    },
    "down": {
     "uv": [
      8.0,
      14.0,
      11.0,
      2.0
     ],
     "cullface": "down",
     "texture": "door_acacia_side"
    },
    "east": {
     "uv": [
      14.0,
      14.0,
      2.0,
      16.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      14.0,
      14.0,
      16.0
     ],
     "cullface": "west",
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    3.0,
    2.0,
    14.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    3.0,
    3.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      13.0,
      13.0,
      14.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      13.0,
      3.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    13.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    3.0,
    14.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    5.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    4.0,
    6.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      5.0,
      12.0,
      6.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      10.0,
      12.0,
      11.0,
      14.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    2.0,
    10.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    4.0,
    11.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      11.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "up": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      4.0,
      12.0,
      5.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      4.0,
      12.0,
      12.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    4.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    4.0,
    12.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    5.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    6.0,
    4.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      10.0,
      4.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      12.0,
      10.0,
      14.0,
      11.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    5.0,
    12.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    6.0,
    14.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    9.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    10.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      9.0,
      0.0,
      10.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      6.0,
      0.0,
      7.0,
      13.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    3.0,
    6.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    7.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    3.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    4.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      12.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      3.0,
      0.0,
      4.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      12.0,
      0.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    12.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    16.0,
    13.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "north": {
     "uv": [
      11.0,
      11.0,
      12.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    11.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    5.0,
    13.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "south": {
     "uv": [
      4.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      11.0,
      11.0,
      13.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      3.0,
      11.0,
      5.0,
      12.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    4.0,
    3.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    5.0,
    5.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      2.0,
      14.0,
      3.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    13.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    14.0,
    14.0
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "down": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "rotation": 90,
     "texture": "door_acacia_lower"
    },
    "east": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "texture": "door_acacia_lower"
    },
    "west": {
     "uv": [
      2.0,
      6.0,
      14.0,
      7.0
     ],
     "texture": "door_acacia_lower"
    }
   },
   "from": [
    1.0,
    9.0,
    2.0
   ],
   "name": "Element",
   "yRot": 180,
   "to": [
    2.0,
    10.0,
    14.0
   ]
  }
 ],
 "71:10": [
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_iron_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "163:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_acacia"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_acacia"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "141:2": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    4,
    -1,
    0
   ],
   "shade": false,
   "to": [
    4,
    15,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    12,
    -1,
    0
   ],
   "shade": false,
   "to": [
    12,
    15,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    0,
    -1,
    4
   ],
   "shade": false,
   "to": [
    16,
    15,
    4
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "carrots_stage_1"
    }
   },
   "from": [
    0,
    -1,
    12
   ],
   "shade": false,
   "to": [
    16,
    15,
    12
   ]
  }
 ],
 "171:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "south",
     "texture": "wool_colored_light_blue"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wool_colored_light_blue"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "wool_colored_light_blue"
    },
    "north": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "north",
     "texture": "wool_colored_light_blue"
    },
    "east": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "east",
     "texture": "wool_colored_light_blue"
    },
    "west": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "west",
     "texture": "wool_colored_light_blue"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    1,
    16
   ]
  }
 ],
 "44:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stonebrick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "stonebrick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stonebrick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stonebrick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stonebrick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stonebrick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  }
 ],
 "147:15": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "gold_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "gold_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "gold_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "81:8": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "cactus_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "cactus_bottom"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    0,
    0,
    1
   ],
   "to": [
    16,
    16,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "cactus_side"
    }
   },
   "from": [
    1,
    0,
    0
   ],
   "to": [
    15,
    16,
    16
   ]
  }
 ],
 "101:0": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "iron_bars"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "iron_bars"
    }
   },
   "from": [
    7,
    0.001,
    7
   ],
   "to": [
    9,
    0.001,
    9
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "iron_bars"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "iron_bars"
    }
   },
   "from": [
    7,
    15.999,
    7
   ],
   "to": [
    9,
    15.999,
    9
   ]
  }
 ],
 "198:2": [
  {
   "faces": {
    "south": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      2,
      6,
      6
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      6,
      6,
      2,
      2
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      2,
      6,
      6,
      7
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    6,
    0,
    6
   ],
   "xRot": 90,
   "yRot": 0,
   "to": [
    10,
    1,
    10
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      2
     ],
     "texture": "end_rod"
    },
    "down": {
     "uv": [
      4,
      2,
      2,
      0
     ],
     "texture": "end_rod"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "east": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    },
    "west": {
     "uv": [
      0,
      0,
      2,
      15
     ],
     "texture": "end_rod"
    }
   },
   "from": [
    7,
    1,
    7
   ],
   "xRot": 90,
   "yRot": 0,
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "157:9": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    2,
    0,
    -0.25
   ],
   "yRot": 90,
   "to": [
    4,
    1,
    16.25
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_activator_powered"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_activator_powered"
    }
   },
   "from": [
    12,
    0,
    -0.25
   ],
   "yRot": 90,
   "to": [
    14,
    1,
    16.25
   ]
  }
 ],
 "195:0": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      16.0,
      11.0,
      13.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      14.0,
      13.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      13.0,
      14.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "to": [
    3.0,
    2.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      15.0
     ],
     "cullface": "west",
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    1.0,
    0.0
   ],
   "name": "Element",
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      13.0,
      11.0,
      8.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      12.0,
      14.0,
      7.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      7.0,
      14.0,
      12.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    7.0
   ],
   "name": "Element",
   "to": [
    3.0,
    2.0,
    12.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      3.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      3.0,
      14.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      14.0,
      3.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    3.0,
    2.0,
    3.0
   ]
  }
 ],
 "120:3": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "endframe_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "end_stone"
    },
    "north": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "east": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    },
    "west": {
     "uv": [
      0,
      3,
      16,
      16
     ],
     "texture": "endframe_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    13,
    16
   ]
  }
 ],
 "26:10": [
  {
   "faces": {
    "east": {
     "uv": [
      15,
      7,
      0,
      16
     ],
     "texture": "bed_head_side"
    },
    "south": {
     "uv": [
      0,
      7,
      16,
      16
     ],
     "texture": "bed_head_end_tall"
    },
    "up": {
     "uv": [
      0,
      0,
      15,
      16
     ],
     "rotation": 90,
     "texture": "bed_head_top"
    },
    "west": {
     "uv": [
      0,
      7,
      15,
      16
     ],
     "texture": "bed_head_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    15
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    0,
    3,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    3,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "texture": "bed_head_end_tall"
    },
    "north": {
     "texture": "bed_head_end_tall"
    },
    "south": {
     "texture": "bed_head_end_tall"
    },
    "up": {
     "uv": [
      0,
      3,
      16,
      4
     ],
     "texture": "bed_head_end_tall"
    },
    "west": {
     "texture": "bed_head_end_tall"
    }
   },
   "from": [
    0,
    0,
    15
   ],
   "yRot": 180,
   "to": [
    16,
    13,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      14,
      3,
      15,
      4
     ],
     "texture": "bed_head_end_tall"
    },
    "north": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "bed_head_end_tall"
    },
    "south": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "bed_head_end_tall"
    },
    "up": {
     "uv": [
      1,
      2,
      15,
      3
     ],
     "texture": "bed_head_end_tall"
    },
    "west": {
     "uv": [
      1,
      3,
      2,
      4
     ],
     "texture": "bed_head_end_tall"
    }
   },
   "from": [
    1,
    13,
    15
   ],
   "yRot": 180,
   "to": [
    15,
    14,
    16
   ]
  }
 ],
 "150:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    4,
    2,
    11
   ],
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    10,
    2,
    11
   ],
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "to": [
    9,
    4,
    4
   ]
  }
 ],
 "55:8": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_dot",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_dust_overlay"
    }
   },
   "from": [
    0,
    0.25,
    0
   ],
   "shade": false,
   "to": [
    16,
    0.25,
    16
   ]
  }
 ],
 "66:8": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "down": {
     "uv": [
      0,
      16,
      16,
      0
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      2,
      15,
      3
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    11,
    0,
    2
   ],
   "yRot": 180,
   "to": [
    16,
    1,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      12,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    2,
    0,
    11
   ],
   "yRot": 180,
   "to": [
    4,
    1,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      12,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    3,
    0,
    9
   ],
   "yRot": 180,
   "to": [
    5,
    1,
    12
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      12,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    9,
    0,
    3
   ],
   "yRot": 180,
   "to": [
    12,
    1,
    5
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    4,
    0,
    7
   ],
   "yRot": 180,
   "to": [
    6,
    1,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    7,
    0,
    4
   ],
   "yRot": 180,
   "to": [
    10,
    1,
    6
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      12,
      3,
      14,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    5,
    0,
    6
   ],
   "yRot": 180,
   "to": [
    7,
    1,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      14,
      3,
      16,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    6,
    0,
    5
   ],
   "yRot": 180,
   "to": [
    8,
    1,
    7
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    12,
    0,
    14
   ],
   "yRot": 180,
   "to": [
    14,
    1,
    16
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    14,
    0,
    12
   ],
   "yRot": 180,
   "to": [
    16,
    1,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "north": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "south": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    },
    "up": {
     "texture": "rail_normal_turned"
    },
    "west": {
     "uv": [
      13,
      3,
      15,
      4
     ],
     "texture": "rail_normal_turned"
    }
   },
   "from": [
    13,
    0,
    13
   ],
   "yRot": 180,
   "to": [
    15,
    1,
    15
   ]
  }
 ],
 "105:2": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      6
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    },
    "south": {
     "uv": [
      16,
      0,
      0,
      6
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    0,
    -1,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    16,
    5,
    8
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      0,
      6
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      6
     ],
     "texture": "melon_stem_disconnected",
     "tintindex": 0
    }
   },
   "from": [
    8,
    -1,
    0
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    5,
    16
   ]
  }
 ],
 "160:12": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_brown"
    },
    "down": {
     "uv": [
      7,
      7,
      9,
      9
     ],
     "texture": "glass_pane_top_brown"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "to": [
    9,
    16,
    9
   ]
  }
 ],
 "127:11": [
  {
   "faces": {
    "south": {
     "uv": [
      7,
      4,
      15,
      13
     ],
     "texture": "cocoa_stage_2"
    },
    "up": {
     "uv": [
      0,
      0,
      7,
      7
     ],
     "texture": "cocoa_stage_2"
    },
    "down": {
     "uv": [
      0,
      0,
      7,
      7
     ],
     "texture": "cocoa_stage_2"
    },
    "north": {
     "uv": [
      7,
      4,
      15,
      13
     ],
     "texture": "cocoa_stage_2"
    },
    "east": {
     "uv": [
      7,
      4,
      15,
      13
     ],
     "texture": "cocoa_stage_2"
    },
    "west": {
     "uv": [
      7,
      4,
      15,
      13
     ],
     "texture": "cocoa_stage_2"
    }
   },
   "from": [
    4,
    3,
    7
   ],
   "yRot": 270,
   "to": [
    12,
    12,
    15
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      16,
      0,
      12,
      4
     ],
     "texture": "cocoa_stage_2"
    },
    "west": {
     "uv": [
      12,
      0,
      16,
      4
     ],
     "texture": "cocoa_stage_2"
    }
   },
   "from": [
    8,
    12,
    12
   ],
   "yRot": 270,
   "to": [
    8,
    16,
    16
   ]
  }
 ],
 "178:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "south",
     "texture": "daylight_detector_side"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "daylight_detector_inverted_top"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "daylight_detector_side"
    },
    "north": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "north",
     "texture": "daylight_detector_side"
    },
    "east": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "east",
     "texture": "daylight_detector_side"
    },
    "west": {
     "uv": [
      0,
      10,
      16,
      16
     ],
     "cullface": "west",
     "texture": "daylight_detector_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    6,
    16
   ]
  }
 ],
 "118:2": [
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    3,
    0
   ],
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_inner"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    2,
    3,
    2
   ],
   "to": [
    14,
    4,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    14,
    3,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "cullface": "north",
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    2,
    3,
    0
   ],
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "cullface": "south",
     "texture": "cauldron_side"
    },
    "up": {
     "cullface": "up",
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_inner"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "cullface": "east",
     "texture": "cauldron_side"
    },
    "west": {
     "cullface": "west",
     "texture": "cauldron_side"
    }
   },
   "from": [
    2,
    3,
    14
   ],
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    4,
    3,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    2
   ],
   "to": [
    2,
    3,
    4
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    12,
    0,
    0
   ],
   "to": [
    16,
    3,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    14,
    0,
    2
   ],
   "to": [
    16,
    3,
    4
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    14
   ],
   "to": [
    4,
    3,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    0,
    0,
    12
   ],
   "to": [
    2,
    3,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    12,
    0,
    14
   ],
   "to": [
    16,
    3,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "cauldron_side"
    },
    "up": {
     "texture": "cauldron_top"
    },
    "down": {
     "texture": "cauldron_bottom"
    },
    "north": {
     "texture": "cauldron_side"
    },
    "east": {
     "texture": "cauldron_side"
    },
    "west": {
     "texture": "cauldron_side"
    }
   },
   "from": [
    14,
    0,
    12
   ],
   "to": [
    16,
    3,
    14
   ]
  },
  {
   "faces": {
    "up": {
     "texture": "water_still"
    }
   },
   "from": [
    2,
    12,
    2
   ],
   "to": [
    14,
    12,
    14
   ]
  }
 ],
 "167:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "iron_trapdoor"
    },
    "up": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "up",
     "texture": "iron_trapdoor"
    },
    "down": {
     "uv": [
      0,
      13,
      16,
      16
     ],
     "cullface": "down",
     "texture": "iron_trapdoor"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "iron_trapdoor"
    },
    "east": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "east",
     "texture": "iron_trapdoor"
    },
    "west": {
     "uv": [
      16,
      0,
      13,
      16
     ],
     "cullface": "west",
     "texture": "iron_trapdoor"
    }
   },
   "from": [
    0,
    0,
    13
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "187:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "yRot": 270,
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "183:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "96:10": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "south",
     "texture": "trapdoor"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "trapdoor"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "trapdoor"
    },
    "north": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "north",
     "texture": "trapdoor"
    },
    "east": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "east",
     "texture": "trapdoor"
    },
    "west": {
     "uv": [
      0,
      16,
      16,
      13
     ],
     "cullface": "west",
     "texture": "trapdoor"
    }
   },
   "from": [
    0,
    13,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "38:2": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_allium"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_allium"
    }
   },
   "from": [
    0.8,
    0,
    8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    15.2,
    16,
    8
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_allium"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "flower_allium"
    }
   },
   "from": [
    8,
    0,
    0.8
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "y",
    "rescale": true
   },
   "to": [
    8,
    16,
    15.2
   ],
   "shade": false
  }
 ],
 "107:12": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_oak"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_oak"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_oak"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "75:5": [
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    9,
    10,
    9
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    0,
    0
   ],
   "shade": false,
   "to": [
    9,
    16,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_off"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    0,
    0,
    7
   ],
   "shade": false,
   "to": [
    16,
    16,
    9
   ]
  }
 ],
 "83:11": [
  {
   "faces": {
    "east": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      6,
      0,
      9,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    7,
    0,
    7
   ],
   "shade": false,
   "to": [
    10,
    16,
    10
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    13,
    -2,
    12
   ],
   "shade": false,
   "to": [
    15,
    14,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    12,
    -1,
    2
   ],
   "shade": false,
   "to": [
    14,
    15,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      13,
      0,
      15,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    2,
    -2,
    2
   ],
   "shade": false,
   "to": [
    4,
    14,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "south": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    },
    "up": {
     "uv": [
      9,
      4,
      11,
      6
     ],
     "texture": "reeds_extra"
    },
    "west": {
     "uv": [
      1,
      0,
      3,
      16
     ],
     "texture": "reeds_extra"
    }
   },
   "from": [
    3,
    0,
    12
   ],
   "shade": false,
   "to": [
    5,
    16,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12.5,
      3,
      15.5,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    12.5,
    9,
    11.5
   ],
   "shade": false,
   "to": [
    15.5,
    11,
    14.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    11.5,
    5,
    1.5
   ],
   "shade": false,
   "to": [
    14.5,
    7,
    4.5
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "up": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "down": {
     "uv": [
      9,
      0,
      12,
      3
     ],
     "texture": "reeds_extra"
    },
    "north": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      0.5,
      8,
      3.5,
      10
     ],
     "texture": "reeds"
    }
   },
   "from": [
    2.5,
    6,
    11.5
   ],
   "shade": false,
   "to": [
    5.5,
    8,
    14.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    }
   },
   "from": [
    10,
    8,
    9
   ],
   "shade": false,
   "to": [
    12,
    10,
    9
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      10,
      13,
      12
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      13,
      10,
      11,
      12
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    2,
    3
   ],
   "shade": false,
   "to": [
    6,
    4,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      3,
      5,
      5
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      5,
      3,
      7,
      5
     ],
     "texture": "reeds"
    }
   },
   "from": [
    5,
    11,
    8
   ],
   "shade": false,
   "to": [
    7,
    13,
    8
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      11,
      1,
      13,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      13,
      1,
      11,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    4,
    8,
    14
   ],
   "shade": false,
   "to": [
    4,
    10,
    16
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    13,
    6,
    1
   ],
   "shade": false,
   "to": [
    13,
    8,
    3
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      11,
      6,
      9,
      8
     ],
     "texture": "reeds"
    },
    "south": {
     "uv": [
      9,
      6,
      11,
      8
     ],
     "texture": "reeds"
    },
    "east": {
     "uv": [
      10,
      1,
      12,
      3
     ],
     "texture": "reeds"
    },
    "west": {
     "uv": [
      12,
      1,
      10,
      3
     ],
     "texture": "reeds"
    }
   },
   "from": [
    3,
    10,
    1
   ],
   "shade": false,
   "to": [
    3,
    12,
    3
   ]
  }
 ],
 "69:8": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "up": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "down": {
     "uv": [
      5,
      4,
      11,
      12
     ],
     "texture": "cobblestone"
    },
    "north": {
     "uv": [
      5,
      0,
      11,
      3
     ],
     "texture": "cobblestone"
    },
    "east": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    },
    "west": {
     "uv": [
      4,
      0,
      12,
      3
     ],
     "texture": "cobblestone"
    }
   },
   "from": [
    5,
    0,
    4
   ],
   "xRot": 180,
   "yRot": 90,
   "to": [
    11,
    3,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "down": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "lever"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      16
     ],
     "texture": "lever"
    }
   },
   "to": [
    9,
    11,
    9
   ],
   "from": [
    7,
    1,
    7
   ],
   "xRot": 180,
   "yRot": 90,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     1,
     8
    ],
    "axis": "x"
   }
  }
 ],
 "28:0": [
  {
   "faces": {
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "rail_detector"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    0.25,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector"
    },
    "up": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "down": {
     "uv": [
      2,
      0,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "north": {
     "uv": [
      12,
      15,
      14,
      16
     ],
     "texture": "rail_detector"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    }
   },
   "from": [
    2,
    0,
    -0.25
   ],
   "to": [
    4,
    1,
    16.25
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "rail_detector"
    },
    "up": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector"
    },
    "down": {
     "uv": [
      12,
      0,
      14,
      16
     ],
     "texture": "rail_detector"
    },
    "north": {
     "uv": [
      2,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "east": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    },
    "west": {
     "uv": [
      3,
      15,
      4,
      16
     ],
     "texture": "rail_detector"
    }
   },
   "from": [
    12,
    0,
    -0.25
   ],
   "to": [
    14,
    1,
    16.25
   ]
  }
 ],
 "163:0": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "planks_acacia"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_acacia"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_acacia"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "planks_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "planks_acacia"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_acacia"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "planks_acacia"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "47:0": [
  {
   "faces": {
    "south": {
     "uv": [
      1.0,
      0.0,
      15.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "up": {
     "uv": [
      1.0,
      1.0,
      15.0,
      15.0
     ],
     "texture": "bookshelf_top"
    },
    "down": {
     "uv": [
      1.0,
      1.0,
      15.0,
      15.0
     ],
     "texture": "bookshelf_top"
    },
    "north": {
     "uv": [
      1.0,
      0.0,
      15.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "east": {
     "uv": [
      1.0,
      0.0,
      15.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "west": {
     "uv": [
      1.0,
      0.0,
      15.0,
      16.0
     ],
     "texture": "bookshelf"
    }
   },
   "from": [
    1.0,
    0.0,
    1.0
   ],
   "name": "Element",
   "to": [
    15.0,
    16.0,
    15.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "up": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "down": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "north": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "east": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "west": {
     "uv": [
      0.0,
      15.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    1.0,
    16.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.0,
      0.0,
      16.0,
      1.0
     ],
     "texture": "bookshelf"
    },
    "up": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "down": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "north": {
     "uv": [
      0.0,
      0.0,
      16.0,
      1.0
     ],
     "texture": "bookshelf"
    },
    "east": {
     "uv": [
      0.0,
      0.0,
      16.0,
      1.0
     ],
     "texture": "bookshelf"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      1.0
     ],
     "texture": "bookshelf"
    }
   },
   "from": [
    0.0,
    15.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      0.0,
      7.0,
      16.0,
      9.0
     ],
     "texture": "bookshelf"
    },
    "up": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "down": {
     "uv": [
      0.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "north": {
     "uv": [
      0.0,
      7.0,
      16.0,
      9.0
     ],
     "texture": "bookshelf"
    },
    "east": {
     "uv": [
      0.0,
      7.0,
      16.0,
      9.0
     ],
     "texture": "bookshelf"
    },
    "west": {
     "uv": [
      0.0,
      7.0,
      16.0,
      9.0
     ],
     "texture": "bookshelf"
    }
   },
   "from": [
    0.0,
    7.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    9.0,
    16.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "east": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "west": {
     "uv": [
      15.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    }
   },
   "from": [
    0.0,
    0.0,
    15.0
   ],
   "name": "Element",
   "to": [
    1.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "east": {
     "uv": [
      15.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    }
   },
   "from": [
    15.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    16.0,
    16.0,
    1.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "south": {
     "uv": [
      15.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "east": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    }
   },
   "from": [
    15.0,
    0.0,
    15.0
   ],
   "name": "Element",
   "to": [
    16.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      15.0,
      0.0,
      16.0,
      16.0
     ],
     "texture": "bookshelf"
    },
    "south": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "east": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf_top"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      1.0,
      16.0
     ],
     "texture": "bookshelf"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "to": [
    1.0,
    16.0,
    1.0
   ]
  }
 ],
 "51:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "154:4": [
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_inside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    10,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    11,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    14,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    0
   ],
   "yRot": 270,
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    14
   ],
   "yRot": 270,
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    4,
    4,
    4
   ],
   "yRot": 270,
   "to": [
    12,
    10,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    6,
    4,
    0
   ],
   "yRot": 270,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "171:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "south",
     "texture": "wool_colored_blue"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "wool_colored_blue"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "wool_colored_blue"
    },
    "north": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "north",
     "texture": "wool_colored_blue"
    },
    "east": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "east",
     "texture": "wool_colored_blue"
    },
    "west": {
     "uv": [
      0,
      15,
      16,
      16
     ],
     "cullface": "west",
     "texture": "wool_colored_blue"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    1,
    16
   ]
  }
 ],
 "197:4": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_dark_oak_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_dark_oak_lower"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_dark_oak_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_dark_oak_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_dark_oak_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 90,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "183:6": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 180,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "148:4": [
  {
   "faces": {
    "south": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "up": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "texture": "iron_block"
    },
    "down": {
     "uv": [
      1,
      1,
      15,
      15
     ],
     "cullface": "down",
     "texture": "iron_block"
    },
    "north": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "east": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    },
    "west": {
     "uv": [
      1,
      15,
      15,
      15.5
     ],
     "texture": "iron_block"
    }
   },
   "from": [
    1,
    0,
    1
   ],
   "to": [
    15,
    0.5,
    15
   ]
  }
 ],
 "106:4": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "vine",
     "tintindex": 0
    }
   },
   "from": [
    0,
    0,
    15.2
   ],
   "yRot": 180,
   "shade": false,
   "to": [
    16,
    16,
    15.2
   ]
  }
 ],
 "114:7": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "south",
     "texture": "nether_brick"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "nether_brick"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "nether_brick"
    },
    "north": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "north",
     "texture": "nether_brick"
    },
    "east": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "east",
     "texture": "nether_brick"
    },
    "west": {
     "uv": [
      0,
      8,
      16,
      16
     ],
     "cullface": "west",
     "texture": "nether_brick"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    8,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "nether_brick"
    },
    "up": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "nether_brick"
    },
    "down": {
     "uv": [
      8,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "nether_brick"
    },
    "north": {
     "uv": [
      0,
      0,
      8,
      8
     ],
     "cullface": "north",
     "texture": "nether_brick"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "nether_brick"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "texture": "nether_brick"
    }
   },
   "from": [
    8,
    8,
    0
   ],
   "xRot": 180,
   "yRot": 270,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "64:9": [
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "door_wood_upper"
    },
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_wood_upper"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_wood_upper"
    },
    "up": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "door_wood_lower"
    },
    "west": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "cullface": "west",
     "texture": "door_wood_upper"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "131:1": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    3.8,
    7.9
   ],
   "yRot": 270,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    9.8,
    4.6,
    11.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.8,
    10.3
   ],
   "yRot": 270,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    8.6,
    4.6,
    10.3
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.8,
    9.1
   ],
   "yRot": 270,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    8.6,
    4.6,
    9.1
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.8,
    9.1
   ],
   "yRot": 270,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    7.4,
    4.6,
    10.3
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    3.8,
    9.1
   ],
   "yRot": 270,
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    8.6,
    4.6,
    10.3
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "yRot": 270,
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "yRot": 270,
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "183:5": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      13,
      2,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      0,
      1,
      2,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    15
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      13,
      16,
      15
     ],
     "texture": "fence_gate_spruce"
    },
    "north": {
     "uv": [
      14,
      1,
      16,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "east": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    13
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    15
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      0,
      9,
      2,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    0,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    2,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      7,
      15,
      10
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    6,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    9,
    13
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "east": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "west": {
     "uv": [
      13,
      1,
      15,
      4
     ],
     "texture": "fence_gate_spruce"
    },
    "up": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    },
    "down": {
     "uv": [
      14,
      9,
      16,
      13
     ],
     "texture": "fence_gate_spruce"
    }
   },
   "from": [
    14,
    12,
    9
   ],
   "yRot": 90,
   "to": [
    16,
    15,
    13
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  }
 ],
 "94:12": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    12
   ],
   "to": [
    9,
    7,
    14
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    11
   ],
   "to": [
    9,
    8,
    15
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    12
   ],
   "to": [
    10,
    8,
    14
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    7,
    2
   ],
   "to": [
    9,
    7,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "to": [
    9,
    8,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "90:0": [
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "portal"
    },
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "portal"
    }
   },
   "from": [
    0,
    0,
    6
   ],
   "to": [
    16,
    16,
    10
   ]
  }
 ],
 "71:3": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
      16
     ],
     "cullface": "north",
     "texture": "door_iron_lower"
    },
    "south": {
     "uv": [
      0,
      0,
      3,
      16
     ],
     "cullface": "south",
     "texture": "door_iron_lower"
    },
    "east": {
     "uv": [
      16,
      0,
      0,
      16
     ],
     "texture": "door_iron_lower"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "west",
     "texture": "door_iron_lower"
    },
    "down": {
     "uv": [
      13,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "door_iron_lower"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    3,
    16,
    16
   ]
  }
 ],
 "154:13": [
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_inside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    10,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    11,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    0,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    2,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    14,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    16,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    0
   ],
   "yRot": 90,
   "to": [
    14,
    16,
    2
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_top"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    2,
    11,
    14
   ],
   "yRot": 90,
   "to": [
    14,
    16,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    4,
    4,
    4
   ],
   "yRot": 90,
   "to": [
    12,
    10,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "texture": "hopper_outside"
    },
    "up": {
     "texture": "hopper_outside"
    },
    "down": {
     "texture": "hopper_outside"
    },
    "north": {
     "texture": "hopper_outside"
    },
    "east": {
     "texture": "hopper_outside"
    },
    "west": {
     "texture": "hopper_outside"
    }
   },
   "from": [
    6,
    4,
    0
   ],
   "yRot": 90,
   "to": [
    10,
    8,
    4
   ]
  }
 ],
 "78:4": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      6,
      16,
      16
     ],
     "cullface": "south",
     "texture": "snow"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "snow"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "snow"
    },
    "north": {
     "uv": [
      0,
      6,
      16,
      16
     ],
     "cullface": "north",
     "texture": "snow"
    },
    "east": {
     "uv": [
      0,
      6,
      16,
      16
     ],
     "cullface": "east",
     "texture": "snow"
    },
    "west": {
     "uv": [
      0,
      6,
      16,
      16
     ],
     "cullface": "west",
     "texture": "snow"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "to": [
    16,
    10,
    16
   ]
  }
 ],
 "29:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "piston_bottom"
    },
    "up": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_inner"
    },
    "east": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "east",
     "texture": "piston_side",
     "rotation": 90
    },
    "west": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "west",
     "texture": "piston_side",
     "rotation": 270
    }
   },
   "from": [
    0,
    0,
    4
   ],
   "xRot": 90,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "93:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "repeater_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 270,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    10
   ],
   "yRot": 270,
   "to": [
    9,
    7,
    12
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "yRot": 270,
   "to": [
    9,
    7,
    4
   ]
  }
 ],
 "29:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "south",
     "texture": "piston_bottom"
    },
    "up": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "up",
     "texture": "piston_side"
    },
    "down": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "down",
     "texture": "piston_side",
     "rotation": 180
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "piston_inner"
    },
    "east": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "east",
     "texture": "piston_side",
     "rotation": 90
    },
    "west": {
     "uv": [
      0,
      4,
      16,
      16
     ],
     "cullface": "west",
     "texture": "piston_side",
     "rotation": 270
    }
   },
   "from": [
    0,
    0,
    4
   ],
   "yRot": 180,
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "126:11": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "south",
     "texture": "planks_jungle"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "up",
     "texture": "planks_jungle"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "planks_jungle"
    },
    "north": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "north",
     "texture": "planks_jungle"
    },
    "east": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "east",
     "texture": "planks_jungle"
    },
    "west": {
     "uv": [
      0,
      0,
      16,
      8
     ],
     "cullface": "west",
     "texture": "planks_jungle"
    }
   },
   "from": [
    0,
    8,
    0
   ],
   "to": [
    16,
    16,
    16
   ]
  }
 ],
 "195:6": [
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      16.0,
      11.0,
      13.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      14.0,
      13.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      13.0,
      14.0,
      16.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    13.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    2.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      0.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      0.0,
      8.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      0.0,
      11.0,
      15.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      16.0,
      0.0,
      0.0,
      15.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      0.0,
      16.0,
      15.0
     ],
     "cullface": "west",
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    1.0,
    0.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    16.0,
    16.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      8.0,
      13.0,
      11.0,
      8.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      12.0,
      14.0,
      7.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      7.0,
      14.0,
      12.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    7.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    2.0,
    12.0
   ]
  },
  {
   "faces": {
    "down": {
     "uv": [
      0.0,
      3.0,
      3.0,
      0.0
     ],
     "texture": "door_jungle_side"
    },
    "north": {
     "uv": [
      11.0,
      14.0,
      8.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "south": {
     "uv": [
      8.0,
      14.0,
      11.0,
      16.0
     ],
     "texture": "door_jungle_side"
    },
    "east": {
     "uv": [
      3.0,
      14.0,
      0.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    },
    "west": {
     "uv": [
      0.0,
      14.0,
      3.0,
      16.0
     ],
     "texture": "door_jungle_lower"
    }
   },
   "from": [
    0.0,
    0.0,
    0.0
   ],
   "name": "Element",
   "yRot": 270,
   "to": [
    3.0,
    2.0,
    3.0
   ]
  }
 ],
 "131:2": [
  {
   "faces": {
    "south": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "up": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "down": {
     "uv": [
      5,
      3,
      11,
      9
     ],
     "texture": "trip_wire_source"
    },
    "north": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "east": {
     "uv": [
      5,
      3,
      11,
      4
     ],
     "texture": "trip_wire_source"
    },
    "west": {
     "uv": [
      5,
      8,
      11,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    6.2,
    3.8,
    7.9
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    9.8,
    4.6,
    11.5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.8,
    10.3
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    8.6,
    4.6,
    10.3
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.8,
    9.1
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    8.6,
    4.6,
    9.1
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      7,
      8,
      9,
      9
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    7.4,
    3.8,
    9.1
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    7.4,
    4.6,
    10.3
   ]
  },
  {
   "faces": {
    "west": {
     "uv": [
      7,
      3,
      9,
      4
     ],
     "texture": "trip_wire_source"
    }
   },
   "from": [
    8.6,
    3.8,
    9.1
   ],
   "rotation": {
    "angle": -45,
    "origin": [
     8,
     6,
     5.2
    ],
    "axis": "x"
   },
   "to": [
    8.6,
    4.6,
    10.3
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      7,
      2,
      9,
      7
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      7,
      9,
      9,
      14
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      7,
      9,
      9,
      11
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      9,
      9,
      14,
      11
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      2,
      9,
      7,
      11
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    7.4,
    5.2,
    10
   ],
   "rotation": {
    "angle": 45,
    "origin": [
     8,
     6,
     14
    ],
    "axis": "x"
   },
   "to": [
    8.8,
    6.8,
    14
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "cullface": "south",
     "texture": "planks_oak"
    },
    "up": {
     "uv": [
      6,
      0,
      10,
      2
     ],
     "texture": "planks_oak"
    },
    "down": {
     "uv": [
      6,
      14,
      10,
      16
     ],
     "texture": "planks_oak"
    },
    "north": {
     "uv": [
      6,
      7,
      10,
      15
     ],
     "texture": "planks_oak"
    },
    "east": {
     "uv": [
      14,
      7,
      16,
      15
     ],
     "texture": "planks_oak"
    },
    "west": {
     "uv": [
      0,
      7,
      2,
      15
     ],
     "texture": "planks_oak"
    }
   },
   "from": [
    6,
    1,
    14
   ],
   "to": [
    10,
    9,
    16
   ]
  }
 ],
 "187:8": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      0,
      7,
      2,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      0,
      0,
      2,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "west",
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    0,
    5,
    7
   ],
   "to": [
    2,
    16,
    9
   ],
   "__comment": "Left-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      14,
      7,
      16,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      14,
      0,
      16,
      11
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "cullface": "east",
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      0,
      9,
      11
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    14,
    5,
    7
   ],
   "to": [
    16,
    16,
    9
   ],
   "__comment": "Right-hand post"
  },
  {
   "faces": {
    "south": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      6,
      7,
      8,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      6,
      1,
      8,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    6,
    6,
    7
   ],
   "to": [
    8,
    15,
    9
   ],
   "__comment": "Inner vertical post of left-hand gate door"
  },
  {
   "faces": {
    "south": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      8,
      7,
      10,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "north": {
     "uv": [
      8,
      1,
      10,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "east": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "west": {
     "uv": [
      7,
      1,
      9,
      10
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    8,
    6,
    7
   ],
   "to": [
    10,
    15,
    9
   ],
   "__comment": "Inner vertical post of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      7,
      6,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    6,
    7
   ],
   "to": [
    6,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      2,
      1,
      6,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      2,
      7,
      6,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    2,
    12,
    7
   ],
   "to": [
    6,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of left-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      7,
      14,
      10
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    6,
    7
   ],
   "to": [
    14,
    9,
    9
   ],
   "__comment": "Lower horizontal bar of right-hand gate door"
  },
  {
   "faces": {
    "north": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "south": {
     "uv": [
      10,
      1,
      14,
      4
     ],
     "texture": "fence_gate_acacia"
    },
    "up": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    },
    "down": {
     "uv": [
      10,
      7,
      14,
      9
     ],
     "texture": "fence_gate_acacia"
    }
   },
   "from": [
    10,
    12,
    7
   ],
   "to": [
    14,
    15,
    9
   ],
   "__comment": "Upper horizontal bar of right-hand gate door"
  }
 ],
 "149:2": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_off"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    4,
    2,
    11
   ],
   "yRot": 180,
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      11
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    10,
    2,
    11
   ],
   "yRot": 180,
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "south": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "down": {
     "uv": [
      7,
      13,
      9,
      15
     ],
     "texture": "redstone_torch_off"
    },
    "north": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "east": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    },
    "west": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_off"
    }
   },
   "from": [
    7,
    2,
    2
   ],
   "yRot": 180,
   "to": [
    9,
    4,
    4
   ]
  }
 ],
 "51:1": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    8.8
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    8.8
   ],
   "shade": false
  },
  {
   "faces": {
    "north": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    0,
    0,
    7.2
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "x",
    "rescale": true
   },
   "to": [
    16,
    22.4,
    7.2
   ],
   "shade": false
  },
  {
   "faces": {
    "west": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    8.8,
    0,
    0
   ],
   "rotation": {
    "angle": -22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    8.8,
    22.4,
    16
   ],
   "shade": false
  },
  {
   "faces": {
    "east": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "fire_layer_0"
    }
   },
   "from": [
    7.2,
    0,
    0
   ],
   "rotation": {
    "angle": 22.5,
    "origin": [
     8,
     8,
     8
    ],
    "axis": "z",
    "rescale": true
   },
   "to": [
    7.2,
    22.4,
    16
   ],
   "shade": false
  }
 ],
 "149:14": [
  {
   "faces": {
    "south": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "south",
     "texture": "stone_slab_top"
    },
    "up": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "texture": "comparator_on"
    },
    "down": {
     "uv": [
      0,
      0,
      16,
      16
     ],
     "cullface": "down",
     "texture": "stone_slab_top"
    },
    "north": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "north",
     "texture": "stone_slab_top"
    },
    "east": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "east",
     "texture": "stone_slab_top"
    },
    "west": {
     "uv": [
      0,
      14,
      16,
      16
     ],
     "cullface": "west",
     "texture": "stone_slab_top"
    }
   },
   "from": [
    0,
    0,
    0
   ],
   "yRot": 180,
   "to": [
    16,
    2,
    16
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    4,
    7,
    11
   ],
   "yRot": 180,
   "to": [
    6,
    7,
    13
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    4,
    2,
    10
   ],
   "yRot": 180,
   "to": [
    6,
    8,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    3,
    2,
    11
   ],
   "yRot": 180,
   "to": [
    7,
    8,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    10,
    7,
    11
   ],
   "yRot": 180,
   "to": [
    12,
    7,
    13
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    10,
    2,
    10
   ],
   "yRot": 180,
   "to": [
    12,
    8,
    14
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      11
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    9,
    2,
    11
   ],
   "yRot": 180,
   "to": [
    13,
    8,
    13
   ]
  },
  {
   "faces": {
    "up": {
     "uv": [
      7,
      6,
      9,
      8
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    5,
    2
   ],
   "yRot": 180,
   "to": [
    9,
    5,
    4
   ]
  },
  {
   "faces": {
    "east": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "west": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    7,
    2,
    1
   ],
   "yRot": 180,
   "to": [
    9,
    6,
    5
   ]
  },
  {
   "faces": {
    "north": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    },
    "south": {
     "uv": [
      6,
      5,
      10,
      9
     ],
     "texture": "redstone_torch_on"
    }
   },
   "from": [
    6,
    2,
    2
   ],
   "yRot": 180,
   "to": [
    10,
    6,
    4
   ]
  }
 ],
 "197:7": [
  {
   "faces": {
    "north": {
     "uv": [
      3,
      0,
      0,
 ]