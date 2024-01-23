#pragma once
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <sys/socket.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "3rdparty/cJSON.h"

class TcpServer{
private:
    int server_fd;
    int opt = 1;
public:
    struct sockaddr_in address;
    int addrlen; 

    TcpServer(int port);
    ~TcpServer();

    int tcpAccept();
};

class TcpAgent{
protected:
    int conn_fd;
    TcpAgent(){}
public:
    struct sockaddr_in address;
    int addrlen; 

    TcpAgent(int _conn_fd, struct sockaddr_in _address);
    ~TcpAgent();

    void tcpSend(void* ptr, size_t size) const;
    void tcpRecv(void* ptr, size_t size) const;
    void tcpSendString(const char* const msg, int msg_len) const;
    const char* tcpRecvString(int& msg_len) const;
    cJSON* tcpRecvJson() const;

};

class TcpClient: public TcpAgent{
public:
    TcpClient(const char* host, int port);
};

class TcpEOFException : public std::exception {};
