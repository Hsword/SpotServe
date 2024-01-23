#include "src/fastertransformer/utils/tcp_utils.h"

std::string getHostIP() {
    struct ifaddrs* ifAddrStruct = nullptr;
    struct ifaddrs* ifa = nullptr;

    if (getifaddrs(&ifAddrStruct) == -1) {
        perror("Error: getifaddrs failed");
        return "";
    }

    for (ifa = ifAddrStruct; ifa != nullptr; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == nullptr) {
            continue;
        }

        if ((ifa->ifa_flags & IFF_LOOPBACK) == 0 && ifa->ifa_addr->sa_family == AF_INET) {
            struct sockaddr_in* sa = reinterpret_cast<struct sockaddr_in*>(ifa->ifa_addr);
            std::string ip = inet_ntoa(sa->sin_addr);
            freeifaddrs(ifAddrStruct);
            return ip;
        }
    }

    if (ifAddrStruct != nullptr) {
        freeifaddrs(ifAddrStruct);
    }

    return "";
}

TcpServer::TcpServer(int port){
    addrlen = sizeof(address);
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        perror("socket failed");
        exit(EXIT_FAILURE);
    }
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
        perror("setsockopt");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons( port );
    if (bind(server_fd, (struct sockaddr *)&address,  sizeof(address)) < 0) {
        perror("bind failed");
        exit(EXIT_FAILURE);
    }
    if (listen(server_fd, 1) < 0) {
        perror("listen");
        exit(EXIT_FAILURE);
    }
}

TcpServer::~TcpServer(){
    close(server_fd);
}

int TcpServer::tcpAccept(){
    int conn_fd;
    if ((conn_fd = accept(server_fd, (struct sockaddr *)&address, (socklen_t*)&addrlen)) < 0) {
        perror("accept");
    }
    return conn_fd;
}

TcpAgent::TcpAgent(int _conn_fd, struct sockaddr_in _address)
    : address(_address)
    , conn_fd(_conn_fd)
    {
        addrlen = sizeof(address);
    }

TcpAgent::~TcpAgent(){
    close(conn_fd);
}

void TcpAgent::tcpSend(void* ptr, size_t size) const{
    int sent_bytes = size;
    void* p = ptr;
    while(sent_bytes > 0) {
        int n = write(conn_fd, p, size);
        if(n < 0){
            perror("write");
            exit(EXIT_FAILURE);
        }

        sent_bytes -= n;
        p = p + n;
    }
}

void TcpAgent::tcpRecv(void* ptr, size_t size) const{
    int remain_bytes = size;
    void* p = ptr;
    while (remain_bytes > 0) {
        int n = read(conn_fd, p, remain_bytes);
        if(n < 0){
            perror("read");
            exit(EXIT_FAILURE);
        }else if(n ==0) break;

        remain_bytes -= n;
        p = p + n;
    }
}

void TcpAgent::tcpSendString(const char* const msg, int msg_len) const{
    tcpSend((void*)(&msg_len), sizeof(int));
    tcpSend((void*)msg, msg_len);
}

const char* TcpAgent::tcpRecvString(int& msg_len) const{
    tcpRecv((void*)(&msg_len), sizeof(int));

    char* str = new char[msg_len + 1];
    tcpRecv((void*)str, msg_len);
    str[msg_len] = '\0';

    return str;
}

void TcpAgent::tcpAsyncSend(void* ptr, size_t size) const{
    int sent_bytes = size;
    void* p = ptr;
    while(sent_bytes > 0) {
        int n = send(conn_fd, p, size, MSG_DONTWAIT);
        if(n < 0){
            perror("write");
            exit(EXIT_FAILURE);
        }

        sent_bytes -= n;
        p = p + n;
    }
}

TcpClient::TcpClient(const char* host, int port){
    if ((conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        printf("\n Socket creation error \n");
        exit(EXIT_FAILURE);
    }
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if(inet_pton(AF_INET, host, &address.sin_addr) <= 0) {
        printf("\nInvalid address/ Address not supported \n");
        exit(EXIT_FAILURE);
    }
    if (connect(conn_fd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        printf("\nConnection Failed \n");
        exit(EXIT_FAILURE);
    }
}
