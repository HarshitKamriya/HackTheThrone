const express = require("express");
const router = express.Router();

const {
  startSession,
  getSessionStatus,
  endSession
} = require("../controllers/sessionController");

router.post("/start", startSession);
router.get("/:sessionId", getSessionStatus);
router.post("/end/:sessionId", endSession);

module.exports = router;