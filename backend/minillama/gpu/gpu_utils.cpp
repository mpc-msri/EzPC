void printAESBlock(uint8_t *b)
{
    for (int i = 0; i < 16; i++)
        printf("%02X", b[i]);
    printf("\n");
}