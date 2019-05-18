import QtQuick 2.11
import QtQuick.Controls 2.5
import QtMultimedia 5.12
import QtWebSockets 1.1

ApplicationWindow {
    visible: true
    width: 640
    height: 480
    title: qsTr("Server")

    function appendMessage(message) {
        var currentDate = new Date();
        var datetime = "[" + currentDate.getHours() + ":"
                + currentDate.getMinutes() + ":" + currentDate.getSeconds() + "]";
        textArea.append(datetime + " " + message);
    }

    WebSocketServer {
        id: server

        property var client: null

        listen: true
        host: "192.168.1.7"
        port: 1234
        onClientConnected: {
            appendMessage("Client connected");
            client = webSocket;
            webSocket.onTextMessageReceived.connect(function(message) {
                appendMessage("Message received from client");
                detector.test_detector(message);
            });
        }
        onErrorStringChanged: {
            appendMessage(qsTr("Server error: %1").arg(errorString));
        }
    }

    Connections {
        target: detector

        onImageRecognized: {
            server.client.sendTextMessage(imageData)
            appendMessage("Message sent to client");
        }
    }

    TextArea {
        id: textArea

        anchors.fill: parent
    }
}
