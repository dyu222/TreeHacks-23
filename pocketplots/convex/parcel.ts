import { query } from "./_generated/server";

const parcelsAll = query(async ({db}) => {
    const parcels = await db.query("parcel").order("asc").collect();
    return parcels;
})