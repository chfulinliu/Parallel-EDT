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