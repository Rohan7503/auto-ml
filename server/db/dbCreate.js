const mongoose = require('mongoose');
const config = require('../config/config');

// Import Mongoose schemas
const users = require('../models/users');

const clientOptions = { 
    serverApi: { version: '1', strict: true, deprecationErrors: true },
    dbName: 'automl'
};
const uri = config.dburi;

async function run() {
    try {
        // Connect to the database
        await mongoose.connect(uri, clientOptions);
        console.log("Connected to the database");

        // Create collections based on schemas
        await Promise.all([
            users.createCollection()
        ]);
        await mongoose.connection.db.admin().command({ ping: 1 });
        console.log("Database created successfully!");
    }
    finally {
        // Ensures that the client will close when you finish/error
        await mongoose.disconnect();
    }
}
run().catch(console.dir);