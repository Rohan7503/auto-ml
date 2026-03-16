// Dummy route - in development

const express = require('express');
const router = express.Router();

const users = require('../models/users');
const authMiddleware = require('../middleware/authMiddleware');

// Homepage route
router.get('/', authMiddleware, async(req, res) => {
    try {
        const user = req.user;

        // Respond with homepage data
        res.status(200).json({message: `Welcome to homepage: ${user.displayName}`});
    } catch(err) {
        // Error handling
        return res.status(500).json({message: err.message});
    }
});

module.exports = router;