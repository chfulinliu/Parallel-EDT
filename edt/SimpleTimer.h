/*
Author: Fulin Liu (fulin.liu@hotmail.com)

File Name: SimpleTimer.h

============================================================================
MIT License

Copyright(c) 2021 Fulin Liu

Permission is hereby granted, free of charge, to any person obtaining a copy
of this softwareand associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright noticeand this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
#pragma once
#include <Windows.h>
#ifdef _WIN32
namespace fl {
	class Timer {
		LARGE_INTEGER _begin;
	public:
		Timer() {
			reset();
		}
		void reset() {
			QueryPerformanceCounter(&_begin);
		}
		double s() const {
			auto t = double(ticks());
			return  t / double(freq().QuadPart);
		}
		double ms() const { return s() * 1e3; }
		double us() const { return s() * 1e6; }
	private:
		static LARGE_INTEGER freq() {
			static LARGE_INTEGER f{};
			if (f.QuadPart == 0)
				QueryPerformanceFrequency(&f);
			return f;
		}
		LONGLONG ticks() const {
			LARGE_INTEGER _end;
			QueryPerformanceCounter(&_end);
			return _end.QuadPart - _begin.QuadPart;
		}
	};
}

#else
#include <chrono>
namespace fl {
	class Timer {
		std::chrono::steady_clock::time_point _begin;
	public:
		Timer() {
			reset();
		}
		void reset() {
			_begin = std::chrono::steady_clock::now();
		}
		double s() const {
			return ns() * 1e-9;
		}
		double ms() const { return ns() * 1e-6; }
		double us() const { return ns() * 1e-3; }

	private:
		size_t ns() const {
			using namespace std::chrono;
			return duration_cast<nanoseconds>(steady_clock::now() - _begin).count();
		}
	};
}
#endif