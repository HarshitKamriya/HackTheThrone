const mongoose = require("mongoose");

const sessionSchema = new mongoose.Schema(
  {
    sessionId: {
      type: String,
      required: true,
      unique: true
    },

    startedAt: {
      type: Date,
      default: Date.now
    },

    endedAt: {
      type: Date,
      expires: 60 * 60 * 24
    },

    status: {
      type: String,
      enum: ["active", "ended"],
      default: "active"
    },

    deviceInfo: {
      type: String
    }
  },
  {
    timestamps: false
  }
);

module.exports = mongoose.model("Session", sessionSchema);
