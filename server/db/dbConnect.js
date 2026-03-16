const mongoose = require('mongoose');
const config = require('../config/config');

const uri = config.dburi;
const clientOptions = {
    serverApi: { version: '1', strict: true, deprecationErrors: true },
    dbName: 'automl'
};

async function run() {
    try {
        await mongoose.connect(uri, clientOptions);
    } catch (err) {
        console.error(err);
    }
}

module.exports = run;