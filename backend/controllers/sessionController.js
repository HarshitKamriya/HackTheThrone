const Session = require("../models/Session");
const crypto = require("crypto");


exports.startSession = async (req, res) => {
  try {
    const sessionId = crypto.randomUUID();

    const session = await Session.create({
      sessionId,
      deviceInfo: req.body.deviceInfo || "unknown"
    });

    res.status(201).json({
      sessionId: session.sessionId,
      status: session.status,
      startedAt: session.startedAt
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};


exports.getSessionStatus = async (req, res) => {
  try {
    const session = await Session.findOne({
      sessionId: req.params.sessionId
    });

    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }

    res.json({
      sessionId: session.sessionId,
      status: session.status,
      startedAt: session.startedAt,
      endedAt: session.endedAt || null
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};


exports.endSession = async (req, res) => {
  try {
    const session = await Session.findOne({
      sessionId: req.params.sessionId
    });

    if (!session) {
      return res.status(404).json({ error: "Session not found" });
    }

    if (session.status === "ended") {
      return res.status(400).json({
        error: "Session already ended"
      });
    }

    session.status = "ended";
    session.endedAt = new Date();

    await session.save();

    res.json({
      sessionId: session.sessionId,
      status: session.status,
      endedAt: session.endedAt
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
};
