package com.xiaomi.taiyi.sdk.transfer.data;

import android.os.Parcel;
import android.os.Parcelable;

/**
 * Client-side stub for the outer AIRequest parcelable used as the bind Intent
 * extra ("request").  Layout is headers + body.
 */
public final class AIRequest implements Parcelable {
    public AIHeaders headers;
    public AIBody body;

    public AIRequest(AIHeaders headers, AIBody body) {
        this.headers = headers;
        this.body = body;
    }

    private AIRequest(Parcel in) {
        readFromParcel(in);
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeParcelable(headers, flags);
        dest.writeParcelable(body, flags);
    }

    public void readFromParcel(Parcel in) {
        headers = in.readParcelable(AIHeaders.class.getClassLoader());
        body = in.readParcelable(AIBody.class.getClassLoader());
    }

    @Override
    public int describeContents() {
        return 0;
    }

    public static final Parcelable.Creator<AIRequest> CREATOR =
            new Parcelable.Creator<AIRequest>() {
                @Override
                public AIRequest createFromParcel(Parcel in) {
                    return new AIRequest(in);
                }

                @Override
                public AIRequest[] newArray(int size) {
                    return new AIRequest[size];
                }
            };
}
