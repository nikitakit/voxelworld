"use strict";

// This function validates input text, and returns an error message if it is
// too short or otherwise malformed.
module.exports = function(text) {
    if (!text || text.length < 10) {
        return "Please enter a longer description. Remember to use complete sentences.";
    }

    // Unconditionally accept long entries.
    if (text.replace(/\s/g, "").length > 40) {
        return null;
    }

    // However, short entries can still be very high quality.
    // We use a heuristic to determine if a short entry looks high-quality and
    // should be accepted without further prompting.
    var numWords = text.match(/\b(\w+)\b/g).length;
    var hasUpperCase = /[a-z]/g.test(text);
    var hasLowerCase = /[A-Z]/g.test(text);
    var hasPunctuation = /[.,!?:;\-()]/g.test(text);
    if (numWords > 6 && hasUpperCase && hasLowerCase && hasPunctuation) {
        return null;
    }

    if (hasUpperCase && hasLowerCase && numWords > 8) {
        return null;
    }

    if (hasPunctuation && numWords > 8) {
        return null;
    }

    return "Please enter a longer description. Remember to use complete sentences.";
}
