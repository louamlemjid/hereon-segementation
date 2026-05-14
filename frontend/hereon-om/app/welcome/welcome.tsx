// Welcome.tsx
import logoDark from "./hereon_logo.svg";
import logoLight from "./hereon_logo.svg";
import heroVideo from "./heroVideo.mp4";
import heroLogoSvg from "./hereon_logo.svg";

export function Welcome() {
  return (
    <main className="relative min-h-screen  text-foreground overflow-hidden">
      {/* Background video */}
      <div className="absolute  ">
        <video
          src={heroVideo}
          autoPlay
          muted
          loop
          playsInline
          className="h-full w-full object-cover"
        >
          Your browser does not support the video tag.
        </video>
        {/* Dark overlay for better text contrast */}
        <div className="absolute inset-0 bg-black/40 backdrop-blur-sm" />
      </div>

      {/* Content container */}
      <div className="relative z-10 container mx-auto px-4 py-16 md:py-24 lg:py-32">
        {/* Hero header with logo */}
        <div className="text-center max-w-3xl mx-auto">
          {/* Logo row */}
          <div className="flex justify-center items-center gap-2 mb-6">
            <img src={heroLogoSvg} alt="Logo" className="h-10 w-auto" />
            <span className="text-primary font-semibold text-xl tracking-tight">
              Mg Alloy Analytics
            </span>
          </div>

          <h1 className="text-4xl md:text-6xl font-bold tracking-tight text-white drop-shadow-lg">
            Segment Any <span className="text-primary">Mg Alloy</span>{" "}
            Microstructure
          </h1>
          <p className="mt-6 text-lg md:text-xl text-white/90 leading-relaxed drop-shadow">
            Instantly isolate grains, phases, and defects in optical microscopy
            images. Powered by next‑generation AI – inspired by Meta SAM 3.
          </p>
          <div className="mt-8 flex flex-wrap justify-center gap-4">
            <button className="bg-primary hover:bg-primary/90 text-white font-medium px-6 py-3 rounded-xl transition-colors shadow-lg">
              Try Segmentation →
            </button>
            <button className="border border-white/50 hover:border-white/80 bg-white/10 backdrop-blur-sm text-white font-medium px-6 py-3 rounded-xl transition-colors">
              Upload Your Image
            </button>
          </div>
        </div>

        {/* Feature highlights – now with semi‑transparent cards */}
        <div className="mt-32 max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-center">
            <div className="p-5 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
              <div className="text-3xl mb-2">🔍</div>
              <h3 className="font-semibold text-white">Instant segmentation</h3>
              <p className="text-sm text-white/80 mt-1">
                Click or text‑prompt any feature – grains, twins, precipitates
              </p>
            </div>
            <div className="p-5 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
              <div className="text-3xl mb-2">⚙️</div>
              <h3 className="font-semibold text-white">Editable outputs</h3>
              <p className="text-sm text-white/80 mt-1">
                Extract masks, measure grain size, or export to CAD
              </p>
            </div>
            <div className="p-5 rounded-xl bg-white/10 backdrop-blur-md border border-white/20 shadow-xl">
              <div className="text-3xl mb-2">📊</div>
              <h3 className="font-semibold text-white">Quantitative analysis</h3>
              <p className="text-sm text-white/80 mt-1">
                Phase fractions, aspect ratios, and defect statistics
              </p>
            </div>
          </div>
        </div>

        {/* Footer CTA */}
        <div className="mt-20 text-center">
          <p className="text-white/70 text-sm drop-shadow">
            Ready to transform your Mg alloy research?{" "}
            <a href="#" className="text-primary hover:underline font-medium">
              Start segmenting now
            </a>
          </p>
        </div>
      </div>
    </main>
  );
}
