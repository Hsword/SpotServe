#pragma once

#include <cstdlib>
#include <map>
#include <vector>
#include <string>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <sys/time.h>

#include "src/fastertransformer/utils/string_utils.h"

namespace fastertransformer {

class Logger {

public:
    enum Level {
        TRACE   = 0,
        DEBUG   = 10,
        INFO    = 20,
        WARNING = 30,
        ERROR   = 40
    };

    static Logger& getLogger()
    {
        static Logger instance;
        return instance;
    }
    Logger(Logger const&)         = delete;
    void operator=(Logger const&) = delete;

    template<typename... Args>
    void log(const Level level, const std::string format, const Args&... args)
    {
        if (level_ <= level) {
            std::string fmt    = getPrefix(level) + format + "\n";
            FILE*       out    = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(out, "%s", logstr.c_str());
        }
    }

    template<typename... Args>
    void log(const Level level, const int rank, const std::string format, const Args&... args)
    {
        if (level_ <= level) {
            std::string fmt    = getPrefix(level, rank) + format + "\n";
            FILE*       out    = level_ < WARNING ? stdout : stderr;
            std::string logstr = fmtstr(fmt, args...);
            fprintf(out, "%s", logstr.c_str());
        }
    }

    void setLevel(const Level level)
    {
        level_ = level;
        log(INFO, "Set logger level by %s", getLevelName(level).c_str());
    }

private:
    const std::string            PREFIX      = "[FT]";
    std::map<Level, std::string> level_name_ = {
        {TRACE, "TRACE"}, {DEBUG, "DEBUG"}, {INFO, "INFO"}, {WARNING, "WARNING"}, {ERROR, "ERROR"}};

#ifndef NDEBUG
    const Level DEFAULT_LOG_LEVEL = DEBUG;
#else
    const Level DEFAULT_LOG_LEVEL = INFO;
#endif
    Level level_ = DEFAULT_LOG_LEVEL;

    Logger()
    {
        char* level_name = std::getenv("FT_LOG_LEVEL");
        if (level_name != nullptr) {
            std::map<std::string, Level> name_to_level = {
                {"TRACE", TRACE},
                {"DEBUG", DEBUG},
                {"INFO", INFO},
                {"WARNING", WARNING},
                {"ERROR", ERROR},
            };
            auto level = name_to_level.find(level_name);
            if (level != name_to_level.end()) {
                setLevel(level->second);
            }
            else {
                fprintf(stderr,
                        "[FT][WARNING] Invalid logger level FT_LOG_LEVEL=%s. "
                        "Ignore the environment variable and use a default "
                        "logging level.\n",
                        level_name);
                level_name = nullptr;
            }
        }
    }

    inline std::string getLevelName(const Level level)
    {
        return level_name_[level];
    }

    inline std::string getPrefix(const Level level)
    {
        struct timeval timestamp;
        gettimeofday(&timestamp, NULL);

        time_t timestamp_secs = timestamp.tv_sec;
        struct tm* timestamp_tm = localtime(&timestamp_secs);
        std::stringstream ss;
        ss << "[" << getLevelName(level) << " " << std::put_time(timestamp_tm, "%Y-%m-%d %H:%M:%S.")
           << std::setfill('0') << std::setw(6) << timestamp.tv_usec << "] ";
        return PREFIX + ss.str();
        // return PREFIX + "[" + getLevelName(level) + fmtstr(" %ld.%06ld", timestamp.tv_sec, timestamp.tv_usec) + "] ";
    }

    inline std::string getPrefix(const Level level, const int rank)
    {
        struct timeval timestamp;
        gettimeofday(&timestamp, NULL);

        time_t timestamp_secs = timestamp.tv_sec;
        struct tm* timestamp_tm = localtime(&timestamp_secs);
        std::stringstream ss;
        ss << "[" << getLevelName(level) << " " << std::put_time(timestamp_tm, "%Y-%m-%d %H:%M:%S.")
           << std::setfill('0') << std::setw(6) << timestamp.tv_usec << "]";
        return PREFIX + ss.str() + "[" + std::to_string(rank) + "] ";
        // return PREFIX + "[" + getLevelName(level) + fmtstr(" %ld.%06ld", timestamp.tv_sec, timestamp.tv_usec) + "][" + std::to_string(rank) + "] ";
    }
};

#define FT_LOG(level, ...) fastertransformer::Logger::getLogger().log(level, __VA_ARGS__)
#define FT_LOG_TRACE(...) FT_LOG(fastertransformer::Logger::TRACE, __VA_ARGS__)
#define FT_LOG_DEBUG(...) FT_LOG(fastertransformer::Logger::DEBUG, __VA_ARGS__)
#define FT_LOG_INFO(...) FT_LOG(fastertransformer::Logger::INFO, __VA_ARGS__)
#define FT_LOG_WARNING(...) FT_LOG(fastertransformer::Logger::WARNING, __VA_ARGS__)
#define FT_LOG_ERROR(...) FT_LOG(fastertransformer::Logger::ERROR, __VA_ARGS__)

class TimeProfiler{
public:
    static TimeProfiler& getProfiler(){
        static TimeProfiler profiler;
        return profiler;
    }

    void append_checkpoint(const char* name, int mask_rank=-1){
        if(mask_rank >= 0 && mask_rank != rank) return;
        struct timeval time;
        gettimeofday(&time, NULL);
        name_list.emplace_back(name);
        time_list.push_back(time);
    }

    void set_rank(int _rank){
        rank = _rank;
    }

    void set_filename(const std::string& _file_name){
        file_name = _file_name;
    }

    void clean_all(){
        name_list.clear();
        time_list.clear();
    }

    void print_all(){
        FILE* file = stdout;
        if(file_name.length() > 0){
            file = fopen(file_name.c_str(), "w");
        }

        fprintf(file, "duration,sum,phase\n");
        if(name_list.size() == 0) return;
        for(unsigned i = 0; i < name_list.size(); i++){
            double t1 = (time_list[i].tv_sec - time_list[0].tv_sec) * 1000 + (time_list[i].tv_usec - time_list[0].tv_usec) * 0.001;
            double t2 = i ? (time_list[i].tv_sec - time_list[i-1].tv_sec) * 1000 + (time_list[i].tv_usec - time_list[i-1].tv_usec) * 0.001 : 0;
            fprintf(file, "%.3f,%.3f,%s\n", t1, t2, name_list[i].c_str());
        }
        clean_all();
        fclose(file);
    }
private:
    int rank = -1;
    std::vector<std::string> name_list;
    std::vector<timeval> time_list;
    std::string file_name;
};

#define FT_PROFILE_ADD(...) fastertransformer::TimeProfiler::getProfiler().append_checkpoint(__VA_ARGS__)
#define FT_PROFILE_SETRANK(rank) fastertransformer::TimeProfiler::getProfiler().set_rank(rank)
#define FT_PROFILE_SETFILE(file_name) fastertransformer::TimeProfiler::getProfiler().set_filename(file_name)
#define FT_PROFILE_PRINT() fastertransformer::TimeProfiler::getProfiler().print_all()
#define FT_PROFILE_CLEAN() fastertransformer::TimeProfiler::getProfiler().clean_all()
}  // namespace fastertransformer
