const express = require('express');
const mongoose = require('mongoose');
const cors = require('cors');
const fs = require('fs');

const dbConnect = require('./db/dbConnect');
const registerRouter = require('./routes/auth/register');
const loginRouter = require('./routes/auth/login');
const homepageRouter = require('./routes/homepage');
const datasetUploadRouter = require('./routes/dataset/upload');
const datasetGetDataframeRouter = require('./routes/dataset/get-dataframe');

const app = express();

//CORS config
const corsOptions = {
    origin: '*',
    methods: ['GET', 'POST', 'PUT', 'DELETE'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true
}
app.use(cors(corsOptions));

// Middleware
app.use(express.json({ limit: '50mb' }));

// Routes
app.use('/auth/register', registerRouter);
app.use('/auth/login', loginRouter);
app.use('/homepage', homepageRouter);
app.use('/dataset/upload', datasetUploadRouter);
app.use('/dataset/get-dataframe', datasetGetDataframeRouter);


// Error handling middleware
app.use((err, req, res, next) => {
    console.error(err.stack);
    res.status(500).send('Something went wrong!');
});


// Connect to the database
dbConnect().catch(console.error);

// Start the server
const PORT = 4000;
const server = app.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
});

// Test the database connection
mongoose.connection.on('error', (error) => {
    console.error('Error connecting to MongoDB:', error);
});

mongoose.connection.once('open', () => {
    console.log('Connected to MongoDB');
});
