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

#ifndef TRUCE_AUX_LIB_AUX_FUNCS_H_
#define TRUCE_AUX_LIB_AUX_FUNCS_H_

#include <stdint.h>
#include <stddef.h>

bool read_all(int fd, uint8_t *buffer, size_t size);
bool write_all(int fd, const uint8_t *buffer, size_t size);

bool inet_listen(int &sockfd, int port);
bool inet_connect(int &sockfd, const char *server_address, int port);
bool inet_accept(int &connfd, int listenfd);

void print_buffer(const uint8_t* buf, int len);

#endif /* TRUCE_AUX_LIB_AUX_FUNCS_H_ */
