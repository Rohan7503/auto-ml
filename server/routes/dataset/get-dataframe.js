const express = require('express');
const router = express.Router();
const axios = require('axios');
const authMiddleware = require('../../middleware/authMiddleware');


// Route to retrieve a dataframe from the flask backend
router.get('/', authMiddleware, async (req, res) => {
    try {
        const response = await axios.get('http://localhost:8000/send_data',
            { headers: { 'Content-Type': 'application/json'} }
        );

        // Send the response back to the client
        res.status(200).json({ data: JSON.parse(response.data) });

    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

module.exports = router;
