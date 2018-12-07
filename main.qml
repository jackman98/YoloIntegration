import QtQuick 2.11
import QtQuick.Window 2.11

Window {
    visible: true
    width: 640
    height: 480
    title: qsTr("Hello World")

    Image {
        anchors.fill: parent
        source: "file:///home/jackman98/build-YoloIntegration-Desktop_Qt_5_11_2_GCC_64bit-Debug/predictions.jpg"
    }
}
