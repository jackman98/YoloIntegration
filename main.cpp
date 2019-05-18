#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QQmlContext>
#include "detector.h"

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;

    SignDetector detector;

    engine.rootContext()->setContextProperty("detector", &detector);

    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    return app.exec();
}
