const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');

const users = require('../../models/users');

// Route to register a new user
router.post('/', async(req, res) => {
    try{
        const displayName = req.body.displayName;
        const email = req.body.email;
        const password = req.body.password;
        
        // Check if all reqd fields are present
        if (!(displayName && email && password))
            return res.status(400).json({message: 'Required field (*) cannot be left blank'});

        // Check if user already exists
        const existingUser = await users.findOne({ email });
        if (existingUser)
            return res.status(409).json({message: 'A user with this email already exists'});

        // Hash the pwd before writing to db
        const hashedPwd = await bcrypt.hash(password, 10);

        // Create a new user object
        const newUser = new users({
            displayName,
            email,
            password: hashedPwd,
        });

        // Write the user object to the db
        const savedUser = await newUser.save();

        // Respond to client w/ success msg
        res.status(201).json({message: 'User registered successfully!'});

    } catch(err){
        // Error handling
        res.status(500).json({message: err.message});
    }
});

module.exports = router;