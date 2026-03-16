const express = require('express');
const router = express.Router();
const multer = require('multer');
const axios = require('axios');
const authMiddleware = require('../../middleware/authMiddleware');

const upload = multer({ storage: multer.memoryStorage() });

// Route to upload a dataset in the flask backend
router.post('/', authMiddleware, upload.single('file'), async (req, res) => {
    try {
        const file = req.file;
        
        // Check if all reqd fields are present
        if (!file)
            return res.status(400).json({ message: 'No file uploaded' });

        // Send the file to the flask backend
        const response = await axios.post('http://localhost:8000/get_data',
            JSON.stringify(file),
            { headers: { 'Content-Type': 'application/json'} }
        );

        // Send the response back to the client
        res.status(200).json({ message: 'File uploaded successfully' });

    } catch (err) {
        res.status(500).json({ message: err.message });
    }
});

module.exports = router;
