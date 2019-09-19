/*
    * Licensed to the Apache Software Foundation (ASF) under one
    * or more contributor license agreements. See the NOTICE file
    * distributed with this work for additional information
    * regarding copyright ownership. The ASF licenses this file
    * to you under the Apache License, Version 2.0 (the
    * "License"); you may not use this file except in compliance
    * with the License. You may obtain a copy of the License at
    *
    * http://www.apache.org/licenses/LICENSE-2.0
    *
    * Unless required by applicable law or agreed to in writing,
    * software distributed under the License is distributed on an
    * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    * KIND, either express or implied. See the License for the
    * specific language governing permissions and limitations
    * under the License.
    */

#include "aux_funcs.h"
#include <stdio.h>
#include <errno.h>
#include <unistd.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>


bool read_all(int fd, uint8_t *buffer, size_t size) {
    int n = 0;
    while (size > 0) {
        n = read(fd, buffer, size);

        if (n < 0) {
            printf("Error: Read failed. Errno = %d\n", errno);
            return false;
        }
        if (n == 0) {
            printf("Reading 0 while left %lu bytes (EOF)\n", size);
            return false;
        }

        size -= n;
        buffer += n;
    }
    return true;
}

bool write_all(int fd, const uint8_t *buffer, size_t size) {
    int n = 0;
    while (size > 0) {
        n = write(fd, buffer, size);
        //printf("In write_all - %c\n", buffer);
        if (n < 0) {
            printf("Error: Write failed. Errno = %d\n", errno);
            return false;
        }
        if (n == 0) {
            printf("Writing 0 while left %lu bytes (EOF)\n", size);
            return false;
        }

        size -= n;
        buffer += n;
    }
    return true;
}

bool inet_listen(int &sockfd, int port) {
    bool retval = false;
    struct sockaddr_in serv_addr;
    int enable = 1;

    errno = 0;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0){
        printf("ERROR: setsockopt(SO_REUSEADDR) failed");
    }
    if(sockfd < 0) {
        printf("ERROR: Could not create socket. errno = %d\n", errno);
        goto cleanup;
    }

    memset(&serv_addr, '0', sizeof(serv_addr));

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    serv_addr.sin_port = htons(port);

    if (bind(sockfd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) != 0) {
        printf("ERROR: bind error occured. errno = %d\n", errno);
        goto cleanup;

    }
    
    if (listen(sockfd, 10) != 0) {
        printf("ERROR: listen error occured. errno = %d\n", errno);
        goto cleanup;
    }
    printf("Listening on port %d\n", port);

    retval = true;

cleanup:

    if (!retval) {
        if (sockfd >= 0) {
            close(sockfd);
            sockfd = -1;
        }
    }

    return retval;

}

bool inet_connect(int &sockfd, const char *server_address, int port) {

    bool retval = false;
    struct sockaddr_in srv_ip_addr;
    int enable = 1;

    errno = 0;
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if(sockfd < 0) {
        printf("ERROR: Could not create socket. errno = %d\n", errno);
        goto cleanup;
    }

    memset(&srv_ip_addr, '0', sizeof(srv_ip_addr));

    srv_ip_addr.sin_family = AF_INET;
    srv_ip_addr.sin_port = htons(port);

    if (inet_pton(AF_INET, server_address, &srv_ip_addr.sin_addr)<=0) {
        printf("ERROR: inet_pton error occured. errno = %d\n", errno);
        goto cleanup;
    }

    if(connect(sockfd, (struct sockaddr *) &srv_ip_addr, sizeof(srv_ip_addr)) < 0) {
       printf("ERROR: Connect Failed. errno = %d\n", errno);
       goto cleanup;
    }
    //if (setsockopt(sockfd, SOL_SOCKET, SO_REUSEADDR, &enable, sizeof(int)) < 0){
    //    printf("ERROR: setsockopt(SO_REUSEADDR) failed");
        //goto cleanup;
    //}
    printf("Connected to %s on port %d\n", server_address, port);

    retval = true;

cleanup:

    if (!retval) {
        if (sockfd >= 0) {
            close(sockfd);
            sockfd = -1;
        }
    }
    return retval;
}



bool inet_accept(int &connfd, int listenfd) {
    bool retval = false;
    struct sockaddr_in addr;
    socklen_t addr_size = sizeof(struct sockaddr_in);
    char ip_str[20] = {0};

    connfd = accept(listenfd, (struct sockaddr*) NULL, NULL);
    if (connfd < 0) {
        printf("ERROR: accept eeror occured. errno = %d\n", errno);
        goto cleanup;
    }
    if (getpeername(connfd, (struct sockaddr *)&addr, &addr_size) != 0) {
        printf("ERROR: getpeername eeror occured. errno = %d\n", errno);
    }
    else {
        strcpy(ip_str, inet_ntoa(addr.sin_addr));
        printf("Connection from %s\n", ip_str);
    }

    retval = true;

cleanup:

    if (!retval) {
        if (connfd >= 0) {
            close(connfd);
            connfd = -1;
        }
    }
    return retval;

}


void print_buffer(const uint8_t* buf, int len)
{
    for (int i=0; i < len; i++) {
        printf("0x%x ", buf[i]);
    }
    printf("\n");
}

