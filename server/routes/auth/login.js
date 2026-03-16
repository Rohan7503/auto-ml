const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');

const users = require('../../models/users');
const config = require('../../config/config');
const secretKey = config.jwtSecret;


// Route to login an existing user
router.post('/', async(req, res) => {
    try{
        const email = req.body.email;
        const password = req.body.password;

        // Check if all reqd fields are present
        if (!(email && password))
            return res.status(400).json({message: 'Required field cannot be left blank'});

        // Ensure user exists
        const user = await users.findOne({ email });
        if (!user)
            return res.status(404).json({message: 'User not found'});

        // Compare pwd with the hashed pwd in the db
        const isPwdValid = user.password ? await bcrypt.compare(password,user.password) : false;
        if (!isPwdValid)
            return res.status(401).json({message: 'Invalid password'});

        // Credentials are valid
        // Generate & sign a JWT and send it to the client
        const token = jwt.sign({_id: user._id}, secretKey);
        res.status(200).json({ token });
    
    } catch(err){
        // Error handling
        return res.status(500).json({message: err.message});
    }
});

module.exports = router;