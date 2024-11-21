#include <barrier>
#include <iostream>
#include <chrono>
#include <thread>
#include <fstream>
using namespace std;

int N, M, n, m;
int **matrice, **kernel, **results, **results2, **results3, **results4;

void secventialDinamic()
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {
            int sum = 0;
            for (int ik = 0; ik < n; ik++)
            {
                for (int jk = 0; jk < m; jk++)
                {
                    int x = i - n / 2 + ik;
                    int y = j - m / 2 + jk;
                    if (x < 0)
                        x = 0;
                    if (y < 0)
                        y = 0;
                    if (x >= N)
                        x = N - 1;
                    if (y >= M)
                        y = M - 1;
                    if (x < 0 && y < 0)
                    {
                        x = 0;
                        y = 0;
                    }
                    if (x < 0 && y >= M)
                    {
                        x = 0;
                        y = M - 1;
                    }
                    if (x >= N && y < 0)
                    {
                        x = N - 1;
                        y = 0;
                    }
                    if (x >= N && y >= M)
                    {
                        x = N - 1;
                        y = M - 1;
                    }
                    sum += matrice[x][y] * kernel[ik][jk];
                }
            }
            results[i][j] = sum;
        }
    }
}

void convolutieInPlace()
{
    int *tempRow = new int[M];
    int *tempRowAbove = new int[M];

    for (int i = 0; i < N; i++)
    {
        // Copy the current row to tempRow
        std::copy(matrice[i], matrice[i] + M, tempRow);

        for (int j = 0; j < M; j++)
        {
            int sum = 0;
            for (int ik = 0; ik < n; ik++)
            {
                for (int jk = 0; jk < m; jk++)
                {
                    int x = i - n / 2 + ik;
                    int y = j - m / 2 + jk;
                    if (x < 0)
                        x = 0;
                    if (y < 0)
                        y = 0;
                    if (x >= N)
                        x = N - 1;
                    if (y >= M)
                        y = M - 1;
                    if (x == i)
                    {
                        sum += tempRow[y] * kernel[ik][jk];
                    }
                    else if (x == i - 1)
                    {
                        sum += tempRowAbove[y] * kernel[ik][jk];
                    }
                    else
                    {
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
            }
            matrice[i][j] = sum;
        }

        // Update tempRowAbove for the next iteration
        if (i < N - 1)
        {
            std::copy(tempRow, tempRow + M, tempRowAbove);
        }
    }

    delete[] tempRow;
    delete[] tempRowAbove;
}

class ThreadLiniiDinamic
{
private:
    int start, stop;

public:
    ThreadLiniiDinamic(int start, int stop)
    {
        this->start = start;
        this->stop = stop;
    }

    void operator()()
    {
        for (int i = start; i < stop; i++)
        {
            for (int j = 0; j < M; j++)
            {
                int sum = 0;
                for (int ik = 0; ik < n; ik++)
                {
                    for (int jk = 0; jk < m; jk++)
                    {
                        int x = i - n / 2 + ik;
                        int y = j - m / 2 + jk;
                        if (x < 0)
                            x = 0;
                        if (y < 0)
                            y = 0;
                        if (x >= N)
                            x = N - 1;
                        if (y >= M)
                            y = M - 1;
                        if (x < 0 && y < 0)
                        {
                            x = 0;
                            y = 0;
                        }
                        if (x < 0 && y >= M)
                        {
                            x = 0;
                            y = M - 1;
                        }
                        if (x >= N && y < 0)
                        {
                            x = N - 1;
                            y = 0;
                        }
                        if (x >= N && y >= M)
                        {
                            x = N - 1;
                            y = M - 1;
                        }
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
                results2[i][j] = sum;
            }
        }
    }
};

class ThreadBlockDinamic
{
private:
    int starti, endi, startj, endj;

public:
    ThreadBlockDinamic(int starti, int endi, int startj, int endj)
    {
        this->starti = starti;
        this->endi = endi;
        this->startj = startj;
        this->endj = endj;
    }
    void operator()()
    {
        for (int i = starti; i < endi; i++)
        {
            int columnStart = (i == starti) ? startj : 0;
            int columnEnd = (i == endi) ? endj : M;
            for (int j = columnStart; j < columnEnd; j++)
            {
                int sum = 0;
                for (int ik = 0; ik < n; ik++)
                {
                    for (int jk = 0; jk < m; jk++)
                    {
                        int x = i - n / 2 + ik;
                        int y = j - m / 2 + jk;
                        if (x < 0)
                            x = 0;
                        if (y < 0)
                            y = 0;
                        if (x >= N)
                            x = N - 1;
                        if (y >= M)
                            y = M - 1;
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
                results4[i][j] = sum;
            }
        }
    }
};

void convolutiiDinamicBlockThread(int nr_threads)
{
    thread **threads = new thread *[nr_threads];

    int starti = 0;
    int endi = 0;
    int startj = 0;
    int endj = 0;

    int linesPerThread = N / nr_threads;
    int rest = N % nr_threads;

    for (int i = 0; i < nr_threads; i++)
    {
        int lines = linesPerThread;
        if (rest > 0)
        {
            lines++;
            rest--;
        }
        endi = starti + lines;
        threads[i] = new thread(ThreadBlockDinamic(starti, endi, startj, endj));
        starti = endi;
    }

    for (int i = 0; i < nr_threads; i++)
    {
        threads[i]->join();
        delete threads[i];
    }
    delete[] threads;
}

void convolutiiDinamicLiniiThread(int nr_threads)
{

    thread **threads = new thread *[nr_threads];

    int start = 0;
    int end = 0;
    int linesPerThread = N / nr_threads;
    int rest = N % nr_threads;

    for (int i = 0; i < nr_threads; i++)
    {
        int lines = linesPerThread;
        if (rest > 0)
        {
            rest--;
            lines += 1;
        }
        end = start + lines;
        threads[i] = new thread(ThreadLiniiDinamic(start, end));
        start = end;
    }
    for (int i = 0; i < nr_threads; i++)
    {
        threads[i]->join();
    }
}

class ThreadInplaceConvolutionTask
{
public:
    ThreadInplaceConvolutionTask(int start, int end, barrier<> &barrier)
        : start(start), end(end), barrier(barrier) {}

    void operator()()
    {
        int numCols = M;
        int *tempRow = new int[numCols];
        int *tempRowAbove1 = new int[numCols];
        int *tempRowBelow1 = new int[numCols];
        int *tempRowAux = new int[numCols];
        int *tempRowCopy = new int[numCols];
        int *tempend1 = new int[numCols];

        if (start > 0)
        {
            std::copy_n(matrice[start - 1], numCols, tempRowAbove1);
        }
        else
        {
            std::copy_n(matrice[start], numCols, tempRowAbove1);
        }
        std::copy_n(matrice[start], numCols, tempRow);
        std::copy_n(matrice[start + 1], numCols, tempend1);
        if (end < N - 1)
        {
            std::copy_n(matrice[end], numCols, tempRowBelow1);
        }
        else
        {
            std::copy_n(matrice[end - 1], numCols, tempRowBelow1);
        }

        barrier.arrive_and_wait();

        for (int i = start; i < end; i++)
        {
            for (int j = 0; j < numCols; j++)
            {
                int sum = 0;
                for (int ki = 0; ki < n; ki++)
                {
                    for (int kj = 0; kj < m; kj++)
                    {
                        int mi = i - n / 2 + ki;
                        int mj = j - m / 2 + kj;
                        mi = std::max(0, std::min(mi, N - 1));
                        mj = std::max(0, std::min(mj, M - 1));
                        if (mi == i)
                        {
                            sum += tempRow[mj] * kernel[ki][kj];
                        }
                        else if (mi == i - 1)
                        {
                            sum += tempRowAbove1[mj] * kernel[ki][kj];
                        }
                        else if (mi == i + 1)
                        {
                            sum += tempend1[mj] * kernel[ki][kj];
                        }
                    }
                }
                tempRowAux[j] = sum;
            }
            std::copy(tempRowAux, tempRowAux + numCols, tempRowCopy);
            std::copy(tempRowCopy, tempRowCopy + numCols, matrice[i]);

            if (i < end - 2)
            {
                std::copy(tempRow, tempRow + numCols, tempRowAbove1);
                std::copy(matrice[i + 1], matrice[i + 1] + numCols, tempRow);
                std::copy(matrice[i + 2], matrice[i + 2] + numCols, tempend1);
            }
            else
            {
                std::copy(tempRow, tempRow + numCols, tempRowAbove1);
                std::copy(tempend1, tempend1 + numCols, tempRow);
                std::copy(tempRowBelow1, tempRowBelow1 + numCols, tempend1);
            }
        }

        delete[] tempRow;
        delete[] tempRowAbove1;
        delete[] tempRowBelow1;
        delete[] tempRowAux;
        delete[] tempRowCopy;
        delete[] tempend1;
    }

private:
    int start;
    int end;
    barrier<> &barrier;
};

void runInplaceConvolutionInThreads(int numThreads)
{

    int chunkSize = N / numThreads;
    int rest = N % numThreads;
    thread **threads = new thread *[numThreads];
    int start = 0;
    int end = 0;

    barrier barr = barrier(numThreads);
    for (int i = 0; i < numThreads; i++)
    {

        int rowsPerThread = chunkSize;
        if (rest > 0)
        {
            rowsPerThread++;
            rest--;
        }
        end = start + rowsPerThread;

        threads[i] = new thread(ThreadInplaceConvolutionTask(start, end, ref(barr)));
        start = end;
    }

    for (int i = 0; i < numThreads; i++)
    {
        threads[i]->join();
    }
}

class ThreadColoaneDinamic
{
private:
    int start, end;

public:
    ThreadColoaneDinamic(int start, int end)
    {
        this->start = start;
        this->end = end;
    }

    void operator()()
    {
        for (int j = start; j < end; j++)
        {
            for (int i = 0; i < N; i++)
            {
                int sum = 0;
                for (int ik = 0; ik < n; ik++)
                {
                    for (int jk = 0; jk < m; jk++)
                    {
                        int x = i - n / 2 + ik;
                        int y = j - m / 2 + jk;
                        if (x < 0)
                            x = 0;
                        if (y < 0)
                            y = 0;
                        if (x >= N)
                            x = N - 1;
                        if (y >= M)
                            y = M - 1;
                        sum += matrice[x][y] * kernel[ik][jk];
                    }
                }
                results3[i][j] = sum;
            }
        }
    }
};

void convolutiiDinamicColoaneThread(int nr_threads)
{
    thread **threads = new thread *[nr_threads];

    int start = 0;
    int end = 0;

    int coloanePerThread = M / nr_threads;
    int rest = M % nr_threads;

    for (int i = 0; i < nr_threads; i++)
    {
        int coloane = coloanePerThread;
        if (rest > 0)
        {
            coloane++;
            rest--;
        }
        end = start + coloane;
        threads[i] = new thread(ThreadColoaneDinamic(start, end));
        start = end;
    }

    for (int i = 0; i < nr_threads; i++)
    {
        threads[i]->join();
    }
}

bool verificaMatrice(int **results, int **results2)
{
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < M; j++)
        {

            if (results[i][j] != results2[i][j])
            {
                return false;
            }
        }
    }
    return true;
}

void appendResultsToFile(const std::string &filename, bool areEqual)
{
    std::ofstream fout(filename, std::ios_base::app);
    if (!fout)
    {
        std::cerr << "Error opening file for writing." << std::endl;
        return;
    }

    fout << "All the files are equal: " << (areEqual ? "True" : "False") << std::endl;
    fout.close();
}

int main(int argc, char *argv[])
{

    ios::sync_with_stdio(false);
    cin.tie(nullptr);
    int nr_threads = atoi(argv[1]);
    string relative_path = argv[2];
    int orientation = atoi(argv[3]);
    string base_path = "--------------------";
    string path = base_path + relative_path;

    ifstream fin(path);
    fin >> N >> M;
    matrice = new int *[N];
    kernel = new int *[N];
    results = new int *[N];
    results4 = new int *[N];
    results2 = new int *[N];
    results3 = new int *[N];
    for (int i = 0; i < N; i++)
    {
        matrice[i] = new int[M];
        results[i] = new int[M];
        results2[i] = new int[M];
        results3[i] = new int[M];
        results4[i] = new int[M];
        for (int j = 0; j < M; j++)
        {
            fin >> matrice[i][j];
        }
    }
    fin >> n >> m;
    kernel = new int *[n];
    for (int i = 0; i < n; i++)
    {
        kernel[i] = new int[m];
        for (int j = 0; j < m; j++)
        {
            fin >> kernel[i][j];
        }
    }
    if (orientation == -2)
    {
        secventialDinamic();

        auto start = chrono::high_resolution_clock::now();
        convolutieInPlace();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);

        cout << duration.count() << endl;

        bool areEqual = verificaMatrice(results, matrice);
        appendResultsToFile("results_dinamic.txt", areEqual);
    }
    else if (orientation == -1)
    {
        auto start = chrono::high_resolution_clock::now();
        secventialDinamic();
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;
    }
    else if (orientation == 0)
    {

        auto start = chrono::high_resolution_clock::now();
        convolutiiDinamicLiniiThread(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;

        secventialDinamic();

        bool areEqual = verificaMatrice(results, results2);
        appendResultsToFile("results_dinamic.txt", areEqual);
    }
    else if (orientation == 1)
    {
        auto start = chrono::high_resolution_clock::now();
        convolutiiDinamicColoaneThread(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;

        secventialDinamic();
        bool areEqual = verificaMatrice(results, results3);
        appendResultsToFile("results_dinamic.txt", areEqual);
    }
    else if (orientation == 2)
    {
        auto start = chrono::high_resolution_clock::now();
        convolutiiDinamicBlockThread(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);
        cout << duration.count() << endl;

        secventialDinamic();
        bool areEqual = verificaMatrice(results, results4);
        appendResultsToFile("results_dinamic.txt", areEqual);
    }
    else if (orientation == 3)
    {
        secventialDinamic();

        auto start = chrono::high_resolution_clock::now();
        runInplaceConvolutionInThreads(nr_threads);
        auto stop = chrono::high_resolution_clock::now();
        auto duration = chrono::duration_cast<chrono::nanoseconds>(stop - start);

        cout << duration.count() << endl;
        bool areEqual = verificaMatrice(results, matrice);
        appendResultsToFile("results_dinamic.txt", areEqual);
    }
}