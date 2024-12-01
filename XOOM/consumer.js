const { Kafka, logLevel } = require('kafkajs');
const fs = require('fs').promises;
const path = require('path');

class KafkaVideoConsumer {
    constructor() {
        // Kafka configuration
        this.kafka = new Kafka({
            clientId: 'video-frames-consumer',
            brokers: ['localhost:9092'],
            retry: {
                initialRetryTime: 100,
                retries: 5
            },
            logLevel: logLevel.INFO
        });

        // Create consumer
        this.consumer = this.kafka.consumer({
            groupId: 'video-consumer-group',
            sessionTimeout: 30000,
            heartbeatInterval: 15000,
            allowAutoTopicCreation: true,
            autoCommit: true, // Keep auto-commit enabled
            autoCommitInterval: 5000 // Commit every 5 seconds
        });

        // Bind methods
        this.connect = this.connect.bind(this);
        this.consume = this.consume.bind(this);
        this.handleMessage = this.handleMessage.bind(this);
    }

    async connect() {
        try {
            // Connect to Kafka
            await this.consumer.connect();
            console.log('Kafka Consumer Connected Successfully');

            // Subscribe to the topic
            await this.consumer.subscribe({
                topic: 'video-frames',
                fromBeginning: true
            });
            console.log('Subscribed to video-frames topic');

            // Start consuming messages
            await this.consume();
        } catch (error) {
            console.error('Kafka Consumer Connection Error:', error);
            this.reconnect();
        }
    }

    async consume() {
        try {
            await this.consumer.run({
                eachMessage: async ({ topic, partition, message }) => {
                    try {
                        await this.handleMessage(topic, partition, message);
                    } catch (error) {
                        console.error('Message Processing Error:', error);
                    }
                }
            });
        } catch (error) {
            console.error('Error while consuming messages:', error);
            this.reconnect();
        }
    }

    async handleMessage(topic, partition, message) {
        try {
            // Convert message value to string
            const messageValue = message.value.toString();

            // Log message details
            console.log('Received Message Details:', {
                topic,
                partition,
                timestamp: new Date().toISOString(),
                offset: message.offset,
                valueLength: messageValue.length
            });

            // Save base64 image
            await this.saveBase64Image(messageValue);
        } catch (error) {
            console.error('Error in message handling:', error);
        }
    }

    async saveBase64Image(base64Data) {
        try {
            // Generate unique filename
            const filename = `image-${Date.now()}.txt`;
            const filePath = path.join(
                process.cwd(),
                'images',
                filename
            );

            // Ensure images directory exists
            await fs.mkdir(path.dirname(filePath), { recursive: true });

            // Write base64 data to file
            await fs.writeFile(filePath, base64Data);
            console.log(`Base64 image saved to: ${filePath}`);
        } catch (error) {
            console.error('Image Saving Error:', error);
        }
    }

    async reconnect() {
        console.log('Attempting to reconnect...');
        try {
            await this.consumer.disconnect();
        } catch (disconnectError) {
            console.error('Error during disconnection:', disconnectError);
        }

        setTimeout(() => {
            this.connect();
        }, 5000);
    }

    async shutdown() {
        try {
            await this.consumer.disconnect();
            console.log('Kafka Consumer Disconnected');
        } catch (error) {
            console.error('Shutdown Error:', error);
        }
    }
}

// Initialize and start the consumer
const videoConsumer = new KafkaVideoConsumer();
videoConsumer.connect();

// Graceful shutdown
process.on('SIGINT', async () => {
    console.log('Shutting down Kafka consumer...');
    await videoConsumer.shutdown();
    process.exit(0);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
    console.error('Uncaught Exception:', error);
});
