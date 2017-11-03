var mcData = require('./mcdata-custom');

module.exports = function(game) {
    return new ChunkConverter(game);
}

function ChunkConverter(game) {
    this.game = game;
    this.stitcher = game.plugins.get('voxel-stitch');
    this.registry = game.plugins.get('voxel-registry');

    const inertBlockProps = mcData.inertBlockProps;
    Object.keys(inertBlockProps).forEach((name) => {
      const props = inertBlockProps[name];

      this.registry.registerBlock(name, props);
    });

    const maxId = 4096; // 2^12 TODO: 2^16? for extended block IDs (plus metadata)

    // array MC block ID -> our block ID
    // packs 4-bit metadata in LSBs (MC block ID = 12-bits, meta = 4-bits, total 16-bits -> ours 16 bit)
    this.translateBlockIDs = new game.arrayType(maxId);
    this.reverseBlockIDs = {};
    this.defaultBlockID = this.registry.getBlockIndex(mcData.mcBlockID2Voxel.default);

    this.mcData = mcData;

    for (var i=0; i < mcData.textureNames.length; i++) {
        this.stitcher.preloadTexture(mcData.textureNames[i]);
    }

    for (let mcID in mcData.mcBlockID2Voxel) {
      let mcBlockID;
      let mcMetaID;
      if (mcID.indexOf(':') !== -1) {
        let a = mcID.split(':');
        mcBlockID = parseInt(a[0], 10);
        mcMetaID = parseInt(a[1], 10);
      } else {
        mcBlockID = parseInt(mcID, 10);
        mcMetaID = 0;
      }
      const ourBlockName = mcData.mcBlockID2Voxel[mcID];
      const ourBlockID = this.registry.getBlockIndex(ourBlockName);
      if (ourBlockID === undefined) {
          console.log("Skipping unrecognized block name", ourBlockName, 'for MC', mcID);
          continue;
        //throw new Error('chunk-converter unrecognized block name: '+ourBlockName+' for MC '+mcID);
      }
      const mcPackedID = (mcMetaID << 8) | mcBlockID;
      if (mcPackedID > maxId) {
          throw new Error('id wraparound');
      }
      this.translateBlockIDs[mcPackedID] = ourBlockID;
      this.reverseBlockIDs[ourBlockID] = mcPackedID;
    }

    // for chunk conversion - see voxel/chunker.js
    this.chunkBits = Math.log(this.game.chunkSize) / Math.log(2); // must be power of two
    this.chunkBits |= 0;
    this.chunkMask = (1 << this.chunkBits) - 1;
}

ChunkConverter.prototype.translateBlockID = function(mcPackedID) {
    var self = this;
    let ourBlockID;
    if (mcPackedID === 0) {
      // air is always air TODO: custom air blocks?
      ourBlockID = 0;
    } else {
      ourBlockID = self.translateBlockIDs[mcPackedID]; // indexed by (4-bit metadata << 8) | (8-bit block ID)
      if (!ourBlockID) ourBlockID = self.translateBlockIDs[mcPackedID & 0xff]; // try 0 metadata
      if (!ourBlockID) {
          //console.log("Found no mapping for block", mcPackedID, "[" + (mcPackedID & 0xff) + ":" + (mcPackedID >> 8) + "]");
          ourBlockID = self.defaultBlockID; // default replacement block
      }
    }

    return ourBlockID;
};
