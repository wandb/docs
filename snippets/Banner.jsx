export const Banner = ({ title, background, children }) => {
  return (
    <div
      className="relative w-full min-h-[200px] bg-gray-900 bg-cover bg-center bg-no-repeat rounded-lg overflow-hidden"
      style={{ backgroundImage: `url('${background}')` }}
    >
      <div className="absolute inset-0 bg-black bg-opacity-40"></div>
      <div className="relative z-10 flex flex-col justify-center h-full px-8 py-12 text-white">
        <h2 className="font-serif text-2xl text-white font-normal mb-4 leading-tight mt-4">{title}</h2>
        <div className="text-gray-200 leading-relaxed">{children}</div>
      </div>
    </div>
  );
};
