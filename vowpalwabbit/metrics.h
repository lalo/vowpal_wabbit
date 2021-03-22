
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include "global_data.h"
#include <cstdio>

using namespace rapidjson;

void Serialize(Writer<FileWriteStream>& writer, const Metrics& metrics) {
    writer.StartObject();
    writer.Key("number_of_events");
    writer.Int(metrics.number_examples_per_pass);
    writer.Key("number_of_features");
    writer.Int(metrics.total_feature_number);
    writer.EndObject();
}

void metrics_to_file(const Metrics& metrics) {
    FILE* fp = fopen("output_metrics.json", "wb"); // non-Windows use "w"
    char writeBuffer[65536];
    FileWriteStream os(fp, writeBuffer, sizeof(writeBuffer));

    Writer<FileWriteStream> writer(os);
    // Metrics m = Metrics();
    Serialize(writer, metrics);
}