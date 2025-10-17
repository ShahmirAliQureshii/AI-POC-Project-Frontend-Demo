"use client";
import { motion } from "framer-motion";

export default function HeroSection({active, setActive}) {
  return (
    <main className="relative min-h-screen overflow-hidden bg-gradient-to-b  via-gray-950 text-black">
      {/* Hero Section */}
      <section className="flex flex-col items-center justify-center h-screen px-6 text-center">
        <motion.h1
          initial={{ opacity: 0, y: 50 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-6xl sm:text-7xl md:text-8xl font-extrabold bg-gradient-to-r from-cyan-400 via-blue-500 to-purple-600 bg-clip-text text-transparent text-black"
        >
          The Future of Intelligence
        </motion.h1>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6, duration: 1 }}
          className="mt-6 text-lg sm:text-xl md:text-2xl max-w-3xl text-black"
        >
          Empowering innovation through Artificial Intelligence.  
          Our platform bridges human creativity with cutting-edge AI technology.
        </motion.p>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 1.2, duration: 1 }}
          className="mt-10 flex gap-6"
        >
          <button className="px-8 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-full text-lg font-semibold hover:scale-105 transition-transform duration-300 shadow-lg">
            Get Started
          </button>
          <button className="px-8 py-3 border border-gray-500 rounded-full text-lg font-semibold hover:bg-gray-800 transition duration-300">
            Learn More
          </button>
        </motion.div>
      </section>

      {/* About Section */}
      <section className="relative z-10 px-6 pb-32 bg-black/40 backdrop-blur-md text-center">
        <motion.h2
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ duration: 1 }}
          className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-400 to-blue-700 bg-clip-text text-transparent"
        >
          About Our Vision
        </motion.h2>
        <p className="mt-6 max-w-3xl mx-auto text-black text-lg leading-relaxed">
          We’re redefining how humans and machines collaborate.  
          From real-time face recognition to intelligent automation —  
          our AI solutions are built for speed, precision, and creativity.
        </p>
      </section>


      {/* Background Glow Effect */}
      <div className="absolute inset-0 -z-10">
        <div className="absolute w-[500px] h-[500px] bg-blue-600/20 blur-[150px] rounded-full top-40 left-10 animate-pulse" />
        <div className="absolute w-[500px] h-[500px] bg-purple-600/20 blur-[150px] rounded-full bottom-20 right-10 animate-pulse" />
      </div>
    </main>
  );
}
