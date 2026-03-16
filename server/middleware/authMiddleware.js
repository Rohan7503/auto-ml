// Authentication middleware to verify the session token

const jwt = require('jsonwebtoken');

const users = require('../models/users');
const config = require('../config/config');
const secretKey = config.jwtSecret;

async function authMiddleware(req, res, next) {
    try {
        const authorization = req.headers.authorization;

        // Check if authorization header is present in request
        if (!authorization)
            return res.status(400).json({message: 'Authorization header missing'});

        const token = authorization.split(' ')[1];

        // Check if token is present in request header
        if (!token)
            return res.status(400).json({message: 'Token missing'});

        // Verify the token
        const decodedToken = jwt.verify(token, secretKey);
        const user = await users.findOne({_id: decodedToken._id});
        if (!user)
            return res.status(401).json({message: 'Unauthorized. Session not found'});

        // Attach the decoded user object to the request object for later use
        req.user = user;
        
        next();
    } catch (err) {
        // Error handling
        res.status(401).json({message: err.message});
    }
}

module.exports = authMiddleware;