const mongoose = require('mongoose');

const usersSchema = new mongoose.Schema({
    displayName: {
        type: String,
        required: true,
    },
    email: {
        type: String,
        required: true,
        unique: true,
    },
    password: {
        type: String,
        required: true,
    },

    // Audit fields
    createdAt: { type: Date, default: Date.now },
    updatedAt: { type: Date, default: Date.now },
});



// Pre-update middleware
usersSchema.pre(/^update/, function(next) {
    this._update.updatedAt = new Date();
    next();
});

const users = mongoose.model('users', usersSchema);

module.exports = users;