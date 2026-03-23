import express from "express";

const router = express.Router();


router.get("/", (req, res) => {
  

  // Redirect user to SSE endpoint
  res.json({
    message: "Nothing to see HERE ! afte update second time"
  });
});

export default router;