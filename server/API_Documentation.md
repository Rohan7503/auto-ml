# API Documentation

## Index
- [API Documentation](#api-documentation)
  - [Index](#index)
  - [Endpoints](#endpoints)
    - [`/auth/register`](#authregister)
    - [`/auth/login`](#authlogin)
    - [`/dataset/upload`](#datasetupload)
    - [`/dataset/get-dataframe`](#datasetget-dataframe)

---

## Endpoints

### `/auth/register`
- **Method:** POST
- **Description:** To register a new user
- **Request Body:**
    - `displayName`  *(required, string)*
    - `email`  *(required, string)*
    - `password`  *(required, string)*
- **Respone:**
    - `201 Created` : User registered successfully  
    **Response object :** 
        - *message*
    - `400 Bad Request` : Invalid request body  
    **Response object :** 
        - *message*
    - `409 Conflict` : Existing email  
    **Response object :** 
        - *message*
    - `500 Internal Server Error`  
    **Response object :** 
        - *message*
---
### `/auth/login`
- **Method:** POST
- **Description:** To log in an existing user
- **Request Body:**
    - `email`  *(required, string)*
    - `password`  *(required, string)*
- **Response:**
    - `200 OK` : Successful login.  
    **Response object :**
        - *token* : JWT session token
    - `400 Bad Request` : Invalid request body  
    **Response object :** 
        - *message*
    - `401 Unauthorized` : Invalid username or password  
    **Response object :** 
        - *message*
    - `404 Not Found` : User not found  
    **Response object :** 
        - *message*
    - `500 Internal Server Error`  
    **Response object :** 
        - *message*
---
### `/dataset/upload`
- **Method:** POST
- **Description:** To upload a dataset in the flask server
- **Request Headers:**
    - `Authorization` *(required, string)* : Bearer Token (JWT token of the session)
- **Request Body: (Multipart/Form-Data)**
    - `file`  *(required, file)* : CSV File
- **Response:**
    - `200 OK` : File successfully uploaded.  
    **Response object :**
        - *message*
    - `400 Bad Request` : Invalid request headers/body  
    **Response object :** 
        - *message*
    - `401 Unauthorized` : Token expired or invalid  
    **Response object :** 
        - *message*
    - `500 Internal Server Error`  
    **Response object :** 
        - *message*
---
### `/dataset/get-dataframe`
- **Method:** GET
- **Description:** To retrieve a dataset from the flask server
- **Request Headers:**
    - `Authorization` *(required, string)* : Bearer Token (JWT token of the session)
- **Response:**
    - `200 OK` : Dataframe retrieved successfully  
    **Response object :**
        - *data (Array)* : Array of the dataframe
    - `400 Bad Request` : Invalid request headers/body  
    **Response object :** 
        - *message*
    - `401 Unauthorized` : Token expired or invalid  
    **Response object :** 
        - *message*
    - `500 Internal Server Error`  
    **Response object :** 
        - *message*
---