#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <include/darknet.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <QImage>

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, float hier_thresh, char *outfile, int fullscreen)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    char **names = get_labels(name_list);

    image **alphabet = load_alphabet();
    network *net = load_network(cfgfile, weightfile, 0);
    set_batch_network(net, 1);
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    float nms=.45;
    if(filename){
        strncpy(input, filename, 256);
    } else {
        printf("Enter Image Path: ");
        fflush(stdout);
        input = "./data/dog.jpg";//fgets(input, 256, stdin);
        if(!input) return;
        strtok(input, "\n");
    }
    image im = load_image_color(input,0,0);
    image sized = letterbox_image(im, net->w, net->h);
    //image sized = resize_image(im, net->w, net->h);
    //image sized2 = resize_max(im, net->w);
    //image sized = crop_image(sized2, -((net->w - sized2.w)/2), -((net->h - sized2.h)/2), net->w, net->h);
    //resize_network(net, sized.w, sized.h);
    layer l = net->layers[net->n-1];


    float *X = sized.data;
    time=what_time_is_it_now();
    network_predict(net, X);
    printf("%s: Predicted in %f seconds.\n", input, what_time_is_it_now()-time);
    int nboxes = 0;
    detection *dets = get_network_boxes(net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes);
    //printf("%d\n", nboxes);
    //if (nms) do_nms_obj(boxes, probs, l.w*l.h*l.n, l.classes, nms);
    if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
    draw_detections(im, dets, nboxes, thresh, names, alphabet, l.classes);
    free_detections(dets, nboxes);
    if(outfile){
        save_image(im, outfile);
    }
    else{
        save_image(im, "predictions");

#ifdef OPENCV
        show_image(im, "predictions", 0);
#endif
    }

    free_image(im);
    free_image(sized);
    return;
}

int main(int argc, char *argv[])
{
    QCoreApplication::setAttribute(Qt::AA_EnableHighDpiScaling);

    QGuiApplication app(argc, argv);
    float thresh = 0.5;
    float hier_thresh = 0.5;

    char *outfile = 0;

    char datacfg[] = "./cfg/coco.data";
    char config[] = "./cfg/yolov3.cfg";
    char weights[] = "./yolov3.weights";


    QQmlApplicationEngine engine;
    engine.load(QUrl(QStringLiteral("qrc:/main.qml")));
    if (engine.rootObjects().isEmpty())
        return -1;

    test_detector(datacfg, config, weights, 0, thresh, hier_thresh, outfile, 0);

    return app.exec();
}
