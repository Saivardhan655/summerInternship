const express = require("express");
const app = express();
const server = require("http").Server(app);
const { v4: uuidv4 } = require("uuid");
const io = require("socket.io")(server);
const { ExpressPeerServer } = require("peer");
const url = require("url");
const peerServer = ExpressPeerServer(server, {
    debug: true,
});
const path = require("path");
const { PythonShell } = require('python-shell');
const bodyParser = require('body-parser');
const fs = require('fs');
const os = require('os');
const { KafkaClient, Producer } = require('kafka-node');

// Middleware
app.use(bodyParser.json({ limit: '50mb' }));
app.use(bodyParser.urlencoded({ limit: '50mb', extended: true }));
app.set("view engine", "ejs");
app.use("/public", express.static(path.join(__dirname, "static")));
app.use("/peerjs", peerServer);

// Utility function for processing frames (placeholder)
function processFrame(frameData, userId) {
    // Implement actual frame processing logic or return a default value
    return 0.2;
}

const kafkaClient = new KafkaClient({ kafkaHost: 'localhost:9092' });
const producer = new Producer(kafkaClient,);

// Routes
app.post("/track-attentiveness", (req, res) => {
    const { userId, roomId, frameData } = req.body;
    
    // Run Python script for attentiveness prediction
    let options = {
        mode: 'json',
        pythonPath: 'python3',
        pythonOptions: ['-u'],
        scriptPath: './ml_scripts/',
        args: [JSON.stringify(frameData), userId, roomId]
    };

    PythonShell.run('attentiveness_predictor.py', options, (err, results) => {
        if (err) {
            console.error("Attentiveness tracking error:", err);
            return res.status(500).json({ error: "Processing failed" });
        }
        
        if (results) {
            // Broadcast attentiveness to room
            io.to(roomId).emit('attentiveness-update', {
                userId: userId,
                attentiveness: results[0].attentiveness
            });
            
            res.status(200).json({ 
                message: "Attentiveness tracked", 
                data: results[0] 
            });
        } else {
            res.status(204).send();
        }
    });
});

app.get("/", (req, res) => {
    res.sendFile(path.join(__dirname, "static", "index.html"));
});

app.get("/join", (req, res) => {
    res.redirect(
        url.format({
            pathname: `/join/${uuidv4()}`,
            query: req.query,
        })
    );
});

app.get("/joinold", (req, res) => {
    res.redirect(
        url.format({
            pathname: req.query.meeting_id,
            query: req.query,
        })
    );
});

app.get("/join/:rooms", (req, res) => {
    res.render("room", { roomid: req.params.rooms, Myname: req.query.name });
});

function sendFrameToKafka(frameData, topic,userId) {
    const payloads = [
        {
            topic: topic,
            messages: JSON.stringify({
                user_id: userId,
                frame_data:frameData
            }),
            partition: 0
        }
    ];

    producer.send(payloads, (err, data) => {
        if (err) {
            console.error('Error sending frame to Kafka:', err);
        } else {
            console.log('Frame sent to Kafka:', data);
        }
    });
}

// Socket.IO Connection Handler
io.on("connection", (socket) => {
    socket.on("join-room", (roomId, id, myname) => {
        socket.join(roomId);
        socket.to(roomId).broadcast.emit("user-connected", id, myname);

        socket.on("messagesend", (message) => {
            console.log(message);
            io.to(roomId).emit("createMessage", message);
        });

        socket.on("tellName", (myname) => {
            console.log(myname);
            socket.to(roomId).broadcast.emit("AddName", myname);
        });

        socket.on("disconnect", () => {
            socket.to(roomId).broadcast.emit("user-disconnected", id);
        });

        socket.on("attentiveness", (data) => {
            // Broadcast attentiveness to room
            io.to(roomId).emit("user-attentiveness", data);
        });

        socket.on("send-frame", (frameData) => {
            try {
                const { userId, frameData: base64Frame, timestamp, roomId } = frameData;
                sendFrameToKafka(base64Frame,'video-frames',userId);
                // Save the base64 image to a file
                saveBase64ToFile(base64Frame, (filePath) => {
                    let options = {
                        mode: 'json',
                        pythonPath: 'python', // Ensure this points to your Python executable
                        pythonOptions: ['-u'],
                        scriptPath: './ml_scripts/', // Path to your Python script
                        args: [filePath] // Pass the file path to Python
                    };

                    PythonShell.run('attentiveness_predictor.py', options, (err, results) => {
                        if (err) {
                            console.error("Frame processing error:", err);
                            return;
                        }
                        
                        if (results) {
                            // Broadcast attentiveness to the room
                            io.to(roomId).emit('attentiveness-update', {
                                userId: userId,
                                attentiveness: results[0].attentiveness
                            });
                        }
                    });
                });
            } catch (error) {
                console.error("Send frame error:", error);
            }
        });

        socket.on("video-frame", (frameData, roomId, userId) => {
            try {
                const attendanceStatus = processFrame(frameData, userId);
                
                // Broadcast updated attendance to the room
                io.to(roomId).emit("attendance-update", {
                    userId: userId,
                    status: attendanceStatus,
                });
            } catch (error) {
                console.error("Video frame processing error:", error);
            }
        });
    });
});

// Utility function to save base64 image data to a file and pass the file path to Python
function saveBase64ToFile(base64Image, callback) {
    // Specify your custom directory (e.g., 'images' folder within your project directory)
    const customDirectory = path.join(__dirname, 'images'); // Change this to your desired folder path

    // Ensure the custom directory exists
    if (!fs.existsSync(customDirectory)) {
        fs.mkdirSync(customDirectory);
    }

    // Create a unique file path within the custom directory
    const tempFilePath = path.join(customDirectory, `image-${Date.now()}.txt`);
    
    // Write the base64 data to the file
    fs.writeFile(tempFilePath, base64Image, 'base64', (err) => {
        if (err) {
            console.error("Error saving base64 data to file:", err);
            return;
        }
        console.log("Base64 image saved to:", tempFilePath);
        callback(tempFilePath);
    });
}

// Start the server
const PORT = process.env.PORT || 3030;
server.listen(PORT, '0.0.0.0', () => {
    console.log(`App is listening on port ${PORT}`);
});