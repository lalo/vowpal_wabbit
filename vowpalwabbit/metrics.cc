
#include "metrics.h"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/filewritestream.h"
#include <cstdio>

using namespace rapidjson;

void Serialize(Writer<FileWriteStream>& writer, const Metrics& metrics) {
    writer.StartObject();
    writer.Key("NumberOfEvents");
    writer.Int(metrics.NumberOfEvents);
    writer.Key("NumberOfLearnedEvents");
    writer.Int(metrics.number_examples_per_pass);
    writer.Key("number_of_features");
    writer.Int(metrics.total_feature_number);
    writer.Key("NumberOfSkippedEvents");
    writer.Int(metrics.NumberOfSkippedEvents);
    writer.Key("FirstEventId");
    writer.String(metrics.FirstEventId.c_str());
    writer.Key("LastEventId");
    writer.String(metrics.LastEventId.c_str());
    writer.Key("NumberOfDanglingObservations");
    writer.Int(metrics.NumberOfDanglingObservations);
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